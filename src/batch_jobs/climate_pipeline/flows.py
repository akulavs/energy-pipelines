"""
Prefect flows for the climate pipeline.

Every flow is parameterized by the request schemas from
``external_data.climate_pipeline.schema`` (so a deployment's run form shows the
full, structured schema for each), exposing only the schemas that flow uses:

- :func:`ingest_era5` -- ``ClimatePipelineRequestArgs`` + ``Era5RequestArgs``.
- :func:`ingest_nsrdb` -- ``ClimatePipelineRequestArgs`` + ``NsrdbRequestArgs``
  (clamped to NSRDB coverage; a range with none is a no-op).
- :func:`era5_nsrdb_silver` -- ``ClimatePipelineRequestArgs``.
- :func:`run_climate_pipeline` -- all three, fanning both bronze sources out
  **concurrently** and then running the silver join as a subflow.

The pipeline does not call the two bronze ingests as subflows: Prefect runs
subflows one after another, so a subflow call can never overlap. Both fan-outs
live in the pipeline flow on its task runner, and the plan/submit/collect helpers
are shared so both paths run the same steps.

All data lands under a single ``root_uri`` (the bronze and silver datasets are
distinguished by their dataset names, not by separate roots). The deployment
points ``root_uri`` at an external location outside the repo; ``DEFAULT_ROOT`` is
only a relative fallback for ad-hoc direct calls.

Follows ``batch_jobs/AGENTS.md``: resolve/compute + write inside tasks, a
flow-manifest lifecycle (start -> COMPLETED / FAILED); the source bronze flows
record no ``input_ids`` (they fetch from an external API).

The two bronze flows fan their fetches out concurrently, bounded by **global
concurrency limits held on the Prefect server** rather than in this file, so the
budgets can be retuned without a deploy. Create them once per environment::

    prefect global-concurrency-limit create cds-api    --limit 5
    prefect global-concurrency-limit create nsrdb-api  --limit 18
    prefect global-concurrency-limit create nsrdb-rate --limit 1000 \
        --slot-decay-per-second 0.27

``nsrdb-rate`` is a *frequency* cap, which a thread pool cannot express -- a pool
bounds how many run at once, never how fast they start. NSRDB publishes 1,000
requests/hour per key (https://developer.nlr.gov/docs/rate-limits/), so the limit
holds 1,000 slots refilling 0.27/s: 972/hour, just under the ceiling.

A limit that does not exist is a no-op, which keeps the offline tests server-free.
"""

from __future__ import annotations

import datetime
import pathlib
import uuid
from collections.abc import Sequence

import prefect
from prefect.concurrency.sync import concurrency, rate_limit
from prefect.task_runners import ThreadPoolTaskRunner

from common.storage.flow_manifest import (
    FlowStatus,
    update_flow_manifest,
    write_flow_manifest_start,
)
from common.storage.manifest import ManifestRow
from external_data.climate_pipeline import (
    dead_letter,
    failures,
    point_manifest,
    schema,
    silver,
)
from external_data.climate_pipeline.era5 import bronze as era5_bronze
from external_data.climate_pipeline.nsrdb import bronze as nsrdb_bronze

# Relative fallback only; deployments override this with an external root_uri.
DEFAULT_ROOT = "climate_pipeline"
DEFAULT_WRITER = "climate_pipeline"

# Provider budgets live on the Prefect server as global concurrency limits, not
# in this file, so they can be retuned without a deploy. Create them once per
# environment (see the module docstring); if they are absent these names are a
# no-op, which is what keeps the offline tests server-free.
#
# Both providers rate-limit *per account*, so more machines would not help.
#
# Provenance differs -- worth knowing before retuning:
#   cds-api (5)    self-imposed. Copernicus publishes no per-user figure and says
#                  its limits vary with load, so there is no number to match.
#   nsrdb-api (18) self-imposed. NSRDB documents no concurrency limit; with the
#                  hourly budget throttling, this is thread-pool backpressure.
#   nsrdb-rate     documented: 1,000 requests/hour per key.
CDS_LIMIT = "cds-api"
NSRDB_LIMIT = "nsrdb-api"
# NSRDB caps request *frequency* as well as count. A thread pool can only cap
# how many run at once, never how fast they start, so the frequency cap needs a
# separate decaying limit.
NSRDB_RATE_LIMIT = "nsrdb-rate"

# Matches ``point_manifest.describe_points`` so both lists truncate alike.
_NAMED_POINT_LIMIT = 5

# Deliberately above the server-side limits so the *limit* is the bottleneck and
# the pool never becomes a second, invisible throttle.
ERA5_MAX_WORKERS = 8
NSRDB_MAX_WORKERS = 24

# Derived from the published hourly budget, not the slot-decay figure: that is how
# the cap is expressed, this is the rate it sustains. Estimation only -- the
# server-side limit is the authority, and each response reports the budget
# actually applied (``nsrdb_bronze.observed_rate_budget``).
NSRDB_SUSTAINED_REQUESTS_PER_SECOND = nsrdb_bronze.DOCUMENTED_REQUESTS_PER_HOUR / 3600

# Timeouts sized from that rate. NSRDB binds: one request per (point, year) at
# 1,000/hour, so wall clock is roughly (points x years) / 1,000 hours -- 27 h for
# 1,000 points of full history, 54 h for 2,000. A thousands-of-points backfill
# therefore does not fit in one run at the documented quota, and no timeout would
# change that. 18 h (~650 points) is the largest run comfortable to supervise;
# split bigger ones across runs, which the manifest skip makes nearly free.
BRONZE_TIMEOUT_SECONDS = 18 * 3600
# Silver is local work -- read one node, join, write -- so it scales with nodes,
# not with provider budget, and needs far less.
SILVER_TIMEOUT_SECONDS = 4 * 3600
# The pipeline overlaps the two bronze fan-outs, so it needs the *larger* of them
# plus silver, not the sum of all three.
PIPELINE_TIMEOUT_SECONDS = BRONZE_TIMEOUT_SECONDS + SILVER_TIMEOUT_SECONDS


def _resolve_root(root_uri: str) -> str:
    """A relative root_uri (e.g. "climate_pipeline") has no scheme; resolve to an
    absolute path so the manifest/parquet writers accept it (mirrors the ingest
    helpers)."""
    if "://" not in str(root_uri):
        return str(pathlib.Path(root_uri).resolve())
    return root_uri


def _flow_prelude() -> tuple[datetime.datetime, datetime.datetime, str]:
    """Common opening values: (run_ts, scheduled_time, flow_id)."""
    run_ts = datetime.datetime.now(tz=datetime.UTC)
    scheduled_time = prefect.runtime.flow_run.scheduled_start_time or run_ts
    flow_id = prefect.runtime.flow_run.id or str(uuid.uuid4())
    return run_ts, scheduled_time, flow_id


def _should_retry(task: object, task_run: object, state: prefect.State) -> bool:
    """
    Retry unless the failure is one that trying again cannot fix.

    Transient is the default: a needlessly retried transient failure costs
    seconds, while treating a transient failure as permanent silently drops a
    point from the run.
    """
    try:
        exc = state.result(raise_on_failure=False)
    except Exception:  # noqa: BLE001 - a state we cannot read is not classifiable
        return True
    return not failures.is_permanent(exc if isinstance(exc, BaseException) else None)


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #


@prefect.task(
    retries=2,
    retry_delay_seconds=[10, 30],
    retry_jitter_factor=0.5,
    # A permanently-failing point burns both retries for nothing, and at scale
    # that budget is the provider's, not ours -- so skip straight to recording it.
    retry_condition_fn=_should_retry,
)
def fetch_era5_point(
    point: tuple[float, float],
    request: schema.ClimatePipelineRequestArgs,
    era5: schema.Era5RequestArgs,
    root_uri: str,
    writer: str,
) -> ManifestRow:
    """
    Fetch one point. Mapped over the request's points so they run concurrently.

    ``cdsapi`` retries internally, so these are only the outer net -- kept low,
    because deep retries at both layers just hammer a rate-limited API. The jitter
    stops a whole fan-out retrying in lockstep after a blip.
    """
    logger = prefect.get_run_logger()
    # Inside the task body, so a retry re-acquires rather than bypassing the cap.
    with concurrency(CDS_LIMIT, occupy=1):
        table = era5_bronze.fetch_point_table(point, request, era5)
    row = era5_bronze.write_point_bronze(
        table, point, request, root_uri, writer, era5=era5
    )
    logger.info(
        "ERA5 (%.4f, %.4f) -> %d row(s) -> %s",
        point[0],
        point[1],
        len(table),
        row.data_uri,
    )
    return row


@prefect.task(
    retries=2,
    retry_delay_seconds=[10, 30],
    retry_jitter_factor=0.5,
    # A permanently-failing point burns both retries for nothing, and at scale
    # that budget is the provider's, not ours -- so skip straight to recording it.
    retry_condition_fn=_should_retry,
)
def fetch_nsrdb_point(
    point: tuple[float, float],
    request: schema.ClimatePipelineRequestArgs,
    nsrdb: schema.NsrdbRequestArgs,
    root_uri: str,
    writer: str,
) -> ManifestRow:
    """
    Fetch one point, year by year. The API unit is a point-*year* -- the CSV
    endpoint serves one of each per request, and ``names=2018,2019`` is rejected
    with ``400 Invalid value(s)``. The *manifest* unit stays the point, so years
    are combined before a single write.

    A **permanently** unavailable year ends the point's range rather than
    discarding it: the years already fetched are written under a key narrowed to
    what they cover (see ``narrow_to_fetched``). Those requests are already spent.

    A **transient** year failure fails the whole point so the task retries. That
    data is expected to arrive, and narrowing on it would make a blip
    indistinguishable from a permanent hole, so no later run would look again.
    """
    logger = prefect.get_run_logger()
    years = nsrdb_bronze.requested_years(request)
    tables = []
    fetched: list[int] = []
    for year in years:
        try:
            # Two caps: how many may be in flight, and how fast they start.
            with concurrency(NSRDB_LIMIT, occupy=1):
                rate_limit(NSRDB_RATE_LIMIT)
                tables.append(nsrdb_bronze.fetch_point_year_table(point, year, nsrdb))
        except Exception as exc:
            # Nothing salvageable, or a retry could still fix it: let the point
            # fail so the retry policy decides, not this loop.
            if not tables or not failures.is_permanent(exc):
                raise
            logger.warning(
                "NSRDB (%.4f, %.4f) %d is permanently unavailable (%s); keeping the "
                "%d year(s) already fetched and ending this point at %d",
                point[0],
                point[1],
                year,
                exc,
                len(tables),
                fetched[-1],
            )
            break
        fetched.append(year)

    # Claim only the span held: a narrowed entry reads as partial coverage, so a
    # later run refetches. The full range would report the gap as covered.
    narrowed = (
        request
        if len(fetched) == len(years)
        else nsrdb_bronze.narrow_to_fetched(request, fetched[-1])
    )
    table = nsrdb_bronze.combine_point_tables(tables, narrowed, len(tables))
    row = nsrdb_bronze.write_point_bronze(
        table, point, narrowed, nsrdb, root_uri, writer
    )
    logger.info(
        "NSRDB (%.4f, %.4f) %d/%d year(s) -> %d row(s) -> %s",
        point[0],
        point[1],
        len(tables),
        len(years),
        table.height,
        row.data_uri,
    )
    return row


def _collect_fetches(
    points: Sequence[tuple[float, float]],
    futures: Sequence[prefect.futures.PrefectFuture],
    source: str,
) -> tuple[list[ManifestRow], list[dict[str, str]]]:
    """
    Wait on a submitted fan-out and split it into written rows and failed points.

    A scattered request always contains some points a provider cannot serve;
    losing a multi-hour run to one is the failure mode this avoids. The caller
    gets both halves and decides, rather than the first exception deciding.

    Takes *futures*, not states: ``.map(..., return_state=True)`` blocks at the
    call, so overlapping two providers needs the lazy handle.
    """
    logger = prefect.get_run_logger()
    rows: list[ManifestRow] = []
    failed: list[dict[str, str]] = []
    for point, future in zip(points, futures, strict=True):
        future.wait()
        state = future.state
        if state.is_completed():
            result = state.result()
            if isinstance(result, ManifestRow):
                rows.append(result)
                continue
            # Completed carrying something else: count it failed rather than
            # drop it, so the point cannot vanish from both lists.
            failed.append(
                dead_letter.record(
                    point,
                    permanent=False,
                    error=f"completed with {type(result).__name__}, not a write",
                )
            )
            continue
        exc = state.result(raise_on_failure=False)
        # str() over repr(): the repo's exceptions render their message there.
        reason = str(exc) if isinstance(exc, BaseException) else str(state.message)
        failed.append(
            dead_letter.record(
                point,
                permanent=failures.is_permanent(
                    exc if isinstance(exc, BaseException) else None
                ),
                error=reason,
            )
        )
        logger.warning("%s (%.4f, %.4f) failed: %s", source, point[0], point[1], reason)
    return rows, failed


# --------------------------------------------------------------------------- #
# Plan / submit / collect
#
# Each bronze source is split into three steps so the *same* code serves both the
# standalone ingest flow and the pipeline. The split exists for the pipeline:
# submitting is what must happen for both providers before either is waited on,
# and only a separated submit makes that expressible.
# --------------------------------------------------------------------------- #


def _plan_era5(
    request: schema.ClimatePipelineRequestArgs,
    root_uri: str,
    era5: schema.Era5RequestArgs,
    *,
    force_refresh: bool,
) -> tuple[list[tuple[float, float]], list[ManifestRow]]:
    """
    The ERA5 points needing a fetch, and the writes already covering the rest.

    Filters on the requested ``variables`` as well as the range, because a write
    holds only what its run fetched: an entry written for a narrower selection
    would otherwise read as coverage, and the variables this run adds would never
    be fetched for that point. A wider stored entry still covers a narrower
    request, so dropping variables re-fetches nothing.
    """
    # De-duplicated to one point per ERA5 node, since points sharing a node
    # download identical data.
    points = era5_bronze.unique_points(request)
    # Ask the manifest what already exists. A point whose stored write covers this
    # range is skipped -- that is what makes adding points, or resuming a killed
    # backfill, cost only the missing work.
    return point_manifest.split_by_coverage(
        points,
        era5_bronze.DATASET_NAME,
        root_uri,
        point_manifest.read_point_key,
        request.start_date,
        request.end_date,
        # Match on nodes, as the write does: a coordinate a few metres from a
        # stored one resolves to the same download and must not re-fetch.
        grain=era5_bronze.node,
        force_refresh=force_refresh,
        match=era5_bronze.variables_match(era5),
    )


def _plan_nsrdb(
    nreq: schema.ClimatePipelineRequestArgs,
    root_uri: str,
    nsrdb: schema.NsrdbRequestArgs,
    *,
    force_refresh: bool,
) -> tuple[list[tuple[float, float]], list[ManifestRow]]:
    """
    The NSRDB points needing a fetch, and the writes already covering the rest.

    Filters on ``interval``, which is part of NSRDB's dataset identity: without it
    a 30-minute write reads as coverage for a 60-minute request, the point is never
    fetched, and silver -- which resolves at 60 -- finds nothing. Re-running would
    not recover, since the planner keeps reporting coverage.

    Filters on the requested ``attributes`` for the same reason a widened ERA5
    selection re-fetches: a write holds only what its run asked for, so a narrower
    entry is not coverage. A wider one still is, so dropping attributes re-fetches
    nothing.
    """
    return point_manifest.split_by_coverage(
        [tuple(p) for p in nreq.points],
        nsrdb_bronze.DATASET_NAME,
        root_uri,
        point_manifest.read_point_key,
        nreq.start_date,
        nreq.end_date,
        force_refresh=force_refresh,
        match=nsrdb_bronze.coverage_match(nsrdb),
    )


def _submit_era5(
    to_fetch: Sequence[tuple[float, float]],
    request: schema.ClimatePipelineRequestArgs,
    era5: schema.Era5RequestArgs,
    root_uri: str,
    writer: str,
) -> list[prefect.futures.PrefectFuture]:
    """
    Start the ERA5 fan-out and return immediately.

    Plain ``.map`` rather than ``return_state=True``, which blocks until every
    point is done -- returning futures is what lets the caller overlap providers.
    """
    if not to_fetch:
        return []
    return list(
        fetch_era5_point.map(
            to_fetch,
            prefect.unmapped(request),
            prefect.unmapped(era5),
            prefect.unmapped(root_uri),
            prefect.unmapped(writer),
        )
    )


def _submit_nsrdb(
    to_fetch: Sequence[tuple[float, float]],
    nreq: schema.ClimatePipelineRequestArgs,
    nsrdb: schema.NsrdbRequestArgs,
    root_uri: str,
    writer: str,
) -> list[prefect.futures.PrefectFuture]:
    """Start the NSRDB fan-out and return immediately; see :func:`_submit_era5`."""
    if not to_fetch:
        return []
    return list(
        fetch_nsrdb_point.map(
            to_fetch,
            prefect.unmapped(nreq),
            prefect.unmapped(nsrdb),
            prefect.unmapped(root_uri),
            prefect.unmapped(writer),
        )
    )


def _warn_if_over_budget(
    to_fetch: Sequence[tuple[float, float]],
    nreq: schema.ClimatePipelineRequestArgs,
    budget_seconds: int,
) -> None:
    """
    Warn when the planned NSRDB work cannot finish inside the flow's timeout.

    Judged on the points actually being fetched, so a resumed backfill is measured
    on what is left. Warns rather than raises: the rate is the published default
    and a key may be granted another, so the estimate can be wrong either way. The
    point is that an oversized run says so at the start, not a day in.
    """
    units = len(to_fetch) * len(nsrdb_bronze.requested_years(nreq))
    if not units:
        return
    estimate = units / NSRDB_SUSTAINED_REQUESTS_PER_SECOND
    logger = prefect.get_run_logger()
    logger.info(
        "NSRDB: %d request(s) at ~%.1f/s -> ~%.1f h of fetching",
        units,
        NSRDB_SUSTAINED_REQUESTS_PER_SECOND,
        estimate / 3600,
    )
    if estimate <= budget_seconds:
        return
    logger.warning(
        "NSRDB: ~%.1f h of fetching exceeds this flow's %.1f h timeout; it will be "
        "cancelled partway. Split the request (a re-run resumes on only the missing "
        "points) or raise the timeout.",
        estimate / 3600,
        budget_seconds / 3600,
    )


def _require_rows(
    source: str,
    rows: Sequence[ManifestRow],
    failed: Sequence[dict[str, str]],
) -> None:
    """
    Fail the run only when a source produced *nothing*.

    Per-run, not per-point: one bad coordinate must not cost a whole fan-out, but
    a source that wrote nothing has not done its job.
    """
    if rows:
        return
    msg = (
        f"no {source} bronze produced: {len(failed)} point(s) failed and none were "
        "already covered"
    )
    raise RuntimeError(msg)


def _log_plan(
    source: str,
    to_fetch: Sequence[tuple[float, float]],
    reusable: Sequence[ManifestRow],
    held: Sequence[dict[str, str]],
    key_reader: point_manifest.KeyReader,
    start: datetime.date,
    end: datetime.date,
) -> None:
    """
    Report the plan by **naming** the points and the range, not only counting.

    "2 already covered, fetching 1" says a point was skipped but not which, so the
    question worth asking needs a dig through the manifest. The requested range
    leads the line and applies throughout, which is what makes each reused point's
    own range worth printing: a wider stored range means reuse from an earlier,
    larger fetch. Truncated past a handful.
    """
    logger = prefect.get_run_logger()
    parts = [f"fetching {len(to_fetch)}"]
    if to_fetch:
        parts[0] += f" [{point_manifest.describe_points(to_fetch)}]"
    if reusable:
        covered = point_manifest.covered_spans(reusable, key_reader)
        parts.append(
            f"{len(reusable)} already covered "
            f"[{point_manifest.describe_coverage(covered)}]"
        )
    if held:
        shown = ", ".join(entry["point"] for entry in held[:_NAMED_POINT_LIMIT])
        if len(held) > _NAMED_POINT_LIMIT:
            shown += f", and {len(held) - _NAMED_POINT_LIMIT} more"
        parts.append(f"{len(held)} recorded dead [{shown}]")
    logger.info("%s %s..%s: %s", source, start, end, "; ".join(parts))


@prefect.task()
def resolve_bronze_inputs(
    request: schema.ClimatePipelineRequestArgs,
    root_uri: str,
    as_of: datetime.datetime | None = None,
) -> tuple[list[ManifestRow], list[ManifestRow], list[str], list[str]]:
    """
    Resolve the bronze writes silver will join, for lineage.

    Bronze is one write per point, so lineage is a *set* of rows per source --
    recording one would name a fraction of what silver was built from. ``as_of``
    bounds the lookup so the recorded ``write_id``s match what the build reads. An
    empty NSRDB list means ERA5-only.

    Missing locations come back twice: all of them for the record, and the subset
    no earlier run explained, which decides whether a gap stops the build.
    """
    logger = prefect.get_run_logger()
    logger.info(
        "resolving bronze inputs under %s for request=%s",
        root_uri,
        request.model_dump_json(),
    )

    era5_by_point, era5_missing = point_manifest.covered_rows(
        era5_bronze.unique_points(request),
        era5_bronze.DATASET_NAME,
        root_uri,
        point_manifest.read_point_key,
        request.start_date,
        request.end_date,
        grain=era5_bronze.node,
        as_of=as_of,
    )
    era5_rows = list(era5_by_point.values())
    if not era5_rows:
        msg = (
            f"no {era5_bronze.DATASET_NAME} for the requested points/range — "
            "ingest the ERA5-Land bronze slice before building silver"
        )
        raise RuntimeError(msg)
    logger.info("resolved %d %s input(s)", len(era5_rows), era5_bronze.DATASET_NAME)

    nreq = silver.nsrdb_request(request)
    if nreq is None:
        logger.info(
            "no NSRDB coverage for the requested range; silver will be ERA5-only"
        )
        return (
            era5_rows,
            [],
            dead_letter.describe({"era5": era5_missing}),
            dead_letter.unexplained(
                root_uri, {"era5": (era5_missing, era5_bronze.node)}
            ),
        )

    # ERA5-Land is hourly, so silver joins the 60-minute NSRDB slice; resolve that
    # specific slice for lineage (interval is part of the NSRDB dataset identity).
    nsrdb_by_point, nsrdb_missing = point_manifest.covered_rows(
        nreq.points,
        nsrdb_bronze.DATASET_NAME,
        root_uri,
        point_manifest.read_point_key,
        nreq.start_date,
        nreq.end_date,
        as_of=as_of,
        match=point_manifest.interval_match(schema.NSRDB_SILVER_INTERVAL),
    )
    nsrdb_rows = list(nsrdb_by_point.values())
    if not nsrdb_rows:
        msg = (
            f"no {nsrdb_bronze.DATASET_NAME} for the requested points/range — "
            "ingest the NSRDB bronze slice before building silver"
        )
        raise RuntimeError(msg)
    logger.info("resolved %d %s input(s)", len(nsrdb_rows), nsrdb_bronze.DATASET_NAME)
    return (
        era5_rows,
        nsrdb_rows,
        dead_letter.describe({"era5": era5_missing, "nsrdb": nsrdb_missing}),
        dead_letter.unexplained(
            root_uri,
            {
                "era5": (era5_missing, era5_bronze.node),
                "nsrdb": (nsrdb_missing, point_manifest.normalise),
            },
        ),
    )


@prefect.task()
def build_silver_table(
    request: schema.ClimatePipelineRequestArgs,
    root_uri: str,
    writer: str,
    as_of: datetime.datetime | None = None,
    allow_missing_points: bool = False,
) -> list[ManifestRow]:
    """
    Build and write the silver grid, one entry per ERA5 node.

    Returns every row written: node-by-node building makes the output a *set* of
    writes, and lineage has to name all of them.
    """
    logger = prefect.get_run_logger()
    logger.info(
        "building silver from bronze under %s for request=%s",
        root_uri,
        request.model_dump_json(),
    )
    rows = silver.ingest_silver(
        request,
        bronze_root=root_uri,
        silver_root=root_uri,
        writer=writer,
        as_of=as_of,
        allow_missing_points=allow_missing_points,
    )
    logger.info("wrote %s -> %d node dataset(s)", silver.DATASET_NAME, len(rows))
    return rows


# --------------------------------------------------------------------------- #
# 1. ERA5 bronze ingestion
# --------------------------------------------------------------------------- #


# prefect's flow overloads omit task_runner, so ty cannot match this call
@prefect.flow(  # ty:ignore[no-matching-overload]
    timeout_seconds=BRONZE_TIMEOUT_SECONDS,
    task_runner=ThreadPoolTaskRunner(max_workers=ERA5_MAX_WORKERS),
)
def ingest_era5(
    *,
    request: schema.ClimatePipelineRequestArgs,
    era5: schema.Era5RequestArgs = schema.Era5RequestArgs(),
    force_refresh: bool = False,
    root_uri: str = DEFAULT_ROOT,
    writer: str = DEFAULT_WRITER,
    as_of: datetime.datetime | None = None,
) -> list[dict[str, str]]:
    """Fetch ERA5-Land timeseries from CDS for ``request`` (points + date range)
    and write the bronze dataset under ``root_uri``. ``era5`` is the fetch config
    (variables). ``as_of`` is accepted for flow-signature consistency but unused:
    this flow fetches from CDS and resolves no inputs.

    Returns the points that failed, so a caller running the whole pipeline can
    tell a gap it just created from one it should refuse to build over."""
    run_ts, scheduled_time, flow_id = _flow_prelude()
    logger = prefect.get_run_logger()
    root_uri = _resolve_root(root_uri)

    write_flow_manifest_start(
        flow_id=flow_id,
        flow_name="ingest_era5",
        writer=writer,
        root_uri=root_uri,
        scheduled_time=scheduled_time,
        start_time=run_ts,
    )
    try:
        to_fetch, reusable = _plan_era5(
            request, root_uri, era5, force_refresh=force_refresh
        )
        to_fetch, held = dead_letter.hold_back(
            to_fetch, root_uri, "era5", era5_bronze.node, force_refresh=force_refresh
        )
        _log_plan(
            "ERA5",
            to_fetch,
            reusable,
            held,
            point_manifest.read_point_key,
            request.start_date,
            request.end_date,
        )
        futures = _submit_era5(to_fetch, request, era5, root_uri, writer)
        fetched, failed = _collect_fetches(to_fetch, futures, "ERA5")
        failed += held
        rows = reusable + fetched
        _require_rows("ERA5", rows, failed)
        if failed:
            logger.warning(
                "ERA5: %d point(s) written, %d failed", len(rows), len(failed)
            )
        update_flow_manifest(
            flow_id=flow_id,
            root_uri=root_uri,
            status=FlowStatus.COMPLETED,
            metadata={
                "output_write_ids": [r.write_id for r in rows],
                # Recorded, not just logged: a run that half-succeeded has to be
                # answerable later about which points it skipped and why.
                "failed_points": failed,
            },
        )
    except Exception as exc:
        update_flow_manifest(
            flow_id=flow_id, root_uri=root_uri, status=FlowStatus.FAILED, error=exc
        )
        raise
    logger.info("Finished ERA5 ingest: %d point dataset(s)", len(rows))
    return failed


# --------------------------------------------------------------------------- #
# 2. NSRDB bronze ingestion
# --------------------------------------------------------------------------- #


# prefect's flow overloads omit task_runner, so ty cannot match this call
@prefect.flow(  # ty:ignore[no-matching-overload]
    timeout_seconds=BRONZE_TIMEOUT_SECONDS,
    task_runner=ThreadPoolTaskRunner(max_workers=NSRDB_MAX_WORKERS),
)
def ingest_nsrdb(
    *,
    request: schema.ClimatePipelineRequestArgs,
    nsrdb: schema.NsrdbRequestArgs = schema.NsrdbRequestArgs(),
    force_refresh: bool = False,
    root_uri: str = DEFAULT_ROOT,
    writer: str = DEFAULT_WRITER,
    as_of: datetime.datetime | None = None,
) -> list[dict[str, str]]:
    """Fetch NSRDB solar timeseries from NREL for ``request`` and write the bronze
    dataset under ``root_uri``. The range is clamped to NSRDB's coverage window; a
    range entirely outside it is a no-op. ``nsrdb`` is the fetch config
    (interval/attributes). ``as_of`` is accepted for flow-signature consistency but
    unused: this flow fetches from NREL and resolves no inputs.

    Returns the points that failed, so a caller running the whole pipeline can
    tell a gap it just created from one it should refuse to build over."""
    run_ts, scheduled_time, flow_id = _flow_prelude()
    logger = prefect.get_run_logger()
    root_uri = _resolve_root(root_uri)
    nreq = silver.nsrdb_request(request)

    write_flow_manifest_start(
        flow_id=flow_id,
        flow_name="ingest_nsrdb",
        writer=writer,
        root_uri=root_uri,
        scheduled_time=scheduled_time,
        start_time=run_ts,
    )
    try:
        if nreq is None:
            logger.warning(
                "requested range %s..%s has no NSRDB coverage; nothing to ingest",
                request.start_date,
                request.end_date,
            )
            update_flow_manifest(
                flow_id=flow_id,
                root_uri=root_uri,
                status=FlowStatus.COMPLETED,
                metadata={"skipped": "no NSRDB coverage for the requested range"},
            )
            return []
        to_fetch, reusable = _plan_nsrdb(
            nreq, root_uri, nsrdb, force_refresh=force_refresh
        )
        to_fetch, held = dead_letter.hold_back(
            to_fetch,
            root_uri,
            "nsrdb",
            point_manifest.normalise,
            force_refresh=force_refresh,
        )
        _log_plan(
            "NSRDB",
            to_fetch,
            reusable,
            held,
            point_manifest.read_point_key,
            nreq.start_date,
            nreq.end_date,
        )
        _warn_if_over_budget(to_fetch, nreq, BRONZE_TIMEOUT_SECONDS)
        futures = _submit_nsrdb(to_fetch, nreq, nsrdb, root_uri, writer)
        fetched, failed = _collect_fetches(to_fetch, futures, "NSRDB")
        failed += held
        rows = reusable + fetched
        # A narrowed write is a gap this run recorded, not a success to pass over.
        failed += dead_letter.short_writes(fetched, nreq.end_date)
        _require_rows("NSRDB", rows, failed)
        if failed:
            logger.warning(
                "NSRDB: %d point(s) written, %d failed", len(rows), len(failed)
            )
        update_flow_manifest(
            flow_id=flow_id,
            root_uri=root_uri,
            status=FlowStatus.COMPLETED,
            metadata={
                "output_write_ids": [r.write_id for r in rows],
                # Recorded, not just logged: a run that half-succeeded has to be
                # answerable later about which points it skipped and why.
                "failed_points": failed,
            },
        )
    except Exception as exc:
        update_flow_manifest(
            flow_id=flow_id, root_uri=root_uri, status=FlowStatus.FAILED, error=exc
        )
        raise
    logger.info("finished NSRDB bronze ingest: %d point dataset(s)", len(rows))
    return failed


# --------------------------------------------------------------------------- #
# 3. Silver join
# --------------------------------------------------------------------------- #


@prefect.flow(timeout_seconds=SILVER_TIMEOUT_SECONDS)
def era5_nsrdb_silver(
    *,
    request: schema.ClimatePipelineRequestArgs,
    root_uri: str = DEFAULT_ROOT,
    writer: str = DEFAULT_WRITER,
    allow_missing_points: bool = False,
    as_of: datetime.datetime | None = None,
) -> None:
    """Read the ERA5 + NSRDB bronze slices for ``request`` from ``root_uri`` and write
    the joined silver grid back under the same ``root_uri``, **one dataset entry per
    ERA5 grid node** -- the join is independent per node, so building and writing
    node by node keeps memory flat however many points were requested. Bronze must
    already be ingested; this only reads and joins. ``as_of`` (default: the scheduled
    time) bounds *both* the lineage resolve and the data read to the same point in
    time, so the recorded ``input_ids`` match the bronze the join actually reads.

    A gap at a point an **earlier** run recorded as permanently unserviceable is
    built over automatically -- that run already established the data will never
    arrive. A gap with no such record still fails, since the usual cause is a
    bronze step never run.

    ``allow_missing_points`` forces the permissive behaviour for *any* gap, and is
    now only needed for a gap nothing recorded."""
    run_ts, scheduled_time, flow_id = _flow_prelude()
    logger = prefect.get_run_logger()
    root_uri = _resolve_root(root_uri)
    resolved_as_of = as_of or scheduled_time

    write_flow_manifest_start(
        flow_id=flow_id,
        flow_name="era5_nsrdb_silver",
        writer=writer,
        root_uri=root_uri,
        scheduled_time=scheduled_time,
        start_time=run_ts,
    )
    try:
        era5_rows, nsrdb_rows, missing, unexplained = resolve_bronze_inputs(
            request, root_uri, resolved_as_of
        )
        # A gap a bronze run accounted for is one to build over; an unexplained
        # gap still stops the build.
        build_over_gaps = allow_missing_points or (bool(missing) and not unexplained)
        if missing and not unexplained and not allow_missing_points:
            logger.warning(
                "%d requested location(s) have no bronze, all recorded permanently "
                "unavailable by an earlier run; building over them: %s",
                len(missing),
                missing,
            )
        # Every per-point write is an input, so lineage names all of them.
        input_ids = [row.write_id for row in (*era5_rows, *nsrdb_rows)]
        resolved = {"era5": [row.write_id for row in era5_rows]}
        if nsrdb_rows:
            resolved["nsrdb"] = [row.write_id for row in nsrdb_rows]
        update_flow_manifest(
            flow_id=flow_id,
            root_uri=root_uri,
            input_ids=input_ids,
            metadata={"resolved_inputs": resolved},
        )

        silver_rows = build_silver_table(
            request, root_uri, writer, resolved_as_of, build_over_gaps
        )
        completed: dict[str, object] = {
            "output_write_ids": [row.write_id for row in silver_rows]
        }
        if missing:
            # A table built over a gap must stay distinguishable from a complete
            # one once the logs roll away.
            completed["missing_points"] = missing
        update_flow_manifest(
            flow_id=flow_id,
            root_uri=root_uri,
            status=FlowStatus.COMPLETED,
            metadata=completed,
        )
    except Exception as exc:
        update_flow_manifest(
            flow_id=flow_id, root_uri=root_uri, status=FlowStatus.FAILED, error=exc
        )
        raise
    logger.info("finished silver join: %d grid node dataset(s)", len(silver_rows))


# --------------------------------------------------------------------------- #
# 4. Full pipeline (all three, in order)
# --------------------------------------------------------------------------- #


# prefect's flow overloads omit task_runner, so ty cannot match this call
@prefect.flow(  # ty:ignore[no-matching-overload]
    timeout_seconds=PIPELINE_TIMEOUT_SECONDS,
    # Both providers' fan-outs share this pool, so it has to hold both budgets at
    # once; the server-side limits, not the pool, stay the real bottleneck.
    task_runner=ThreadPoolTaskRunner(max_workers=ERA5_MAX_WORKERS + NSRDB_MAX_WORKERS),
)
def run_climate_pipeline(
    *,
    request: schema.ClimatePipelineRequestArgs,
    era5: schema.Era5RequestArgs = schema.Era5RequestArgs(),
    nsrdb: schema.NsrdbRequestArgs = schema.NsrdbRequestArgs(),
    root_uri: str = DEFAULT_ROOT,
    writer: str = DEFAULT_WRITER,
    force_refresh: bool = False,
    allow_missing_points: bool = False,
    as_of: datetime.datetime | None = None,
) -> None:
    """
    End-to-end: ingest both bronze sources **concurrently**, then build the silver
    grid. All datasets land under ``root_uri``.

    The two providers have independent per-account limits, so both fan-outs are
    submitted before either is waited on and they run in their own rate lanes. The
    saving is whichever source finishes first -- hours on a large backfill.

    The fetches are fanned out **here** rather than by calling the bronze flows as
    subflows, because Prefect runs subflows one after another and a subflow call
    cannot overlap. Those flows remain the standalone entry points and share every
    step with this path via the plan/submit/collect helpers.

    Since the fetches are this flow's own task runs, the run records **one**
    flow-manifest entry with ``output_write_ids`` and ``failed_points`` keyed by
    source. Synthesising per-source entries would record flow runs that never
    happened.

    ``force_refresh`` refetches instead of reusing covered points. A range outside
    NSRDB coverage ingests ERA5 only. ``nsrdb.interval`` is validated up front,
    since silver reads the 60-minute slice.
    """
    run_ts, scheduled_time, flow_id = _flow_prelude()
    logger = prefect.get_run_logger()
    root_uri = _resolve_root(root_uri)

    write_flow_manifest_start(
        flow_id=flow_id,
        flow_name="run_climate_pipeline",
        writer=writer,
        root_uri=root_uri,
        scheduled_time=scheduled_time,
        start_time=run_ts,
    )
    try:
        logger.info(
            "input params -> request=%s era5=%s nsrdb=%s",
            request.model_dump_json(),
            era5.model_dump_json(),
            nsrdb.model_dump_json(),
        )

        # Fail fast: silver reads the 60-minute slice, so any other interval would
        # fetch both sources and only then fail. Validate before fetching.
        if nsrdb.interval != schema.NSRDB_SILVER_INTERVAL:
            msg = (
                f"run_climate_pipeline needs NSRDB interval="
                f"{schema.NSRDB_SILVER_INTERVAL} because the silver join reads the "
                f"hourly slice, but the NSRDB config has interval={nsrdb.interval}. "
            )
            logger.error(msg)
            raise RuntimeError(msg)

        # Plan both first: cheap lookups, and knowing the work list up front is
        # what lets both fan-outs start together.
        nreq = silver.nsrdb_request(request)
        era5_to_fetch, era5_reusable = _plan_era5(
            request, root_uri, era5, force_refresh=force_refresh
        )
        era5_to_fetch, era5_held = dead_letter.hold_back(
            era5_to_fetch,
            root_uri,
            "era5",
            era5_bronze.node,
            force_refresh=force_refresh,
        )
        nsrdb_to_fetch: list[tuple[float, float]] = []
        nsrdb_reusable: list[ManifestRow] = []
        nsrdb_held: list[dict[str, str]] = []
        if nreq is None:
            logger.warning(
                "requested range %s..%s has no NSRDB coverage; ingesting ERA5 only",
                request.start_date,
                request.end_date,
            )
        else:
            nsrdb_to_fetch, nsrdb_reusable = _plan_nsrdb(
                nreq, root_uri, nsrdb, force_refresh=force_refresh
            )
            nsrdb_to_fetch, nsrdb_held = dead_letter.hold_back(
                nsrdb_to_fetch,
                root_uri,
                "nsrdb",
                point_manifest.normalise,
                force_refresh=force_refresh,
            )
        logger.info("step 1/2: bronze ingest, both sources concurrently")
        _log_plan(
            "ERA5",
            era5_to_fetch,
            era5_reusable,
            era5_held,
            point_manifest.read_point_key,
            request.start_date,
            request.end_date,
        )
        _log_plan(
            "NSRDB",
            nsrdb_to_fetch,
            nsrdb_reusable,
            nsrdb_held,
            point_manifest.read_point_key,
            # The clamped NSRDB window, not the joined request: that is the range
            # actually being fetched when the request runs past NSRDB coverage.
            nreq.start_date if nreq else request.start_date,
            nreq.end_date if nreq else request.end_date,
        )

        if nreq is not None:
            _warn_if_over_budget(nsrdb_to_fetch, nreq, PIPELINE_TIMEOUT_SECONDS)

        # Submit both before waiting on either -- this ordering *is* the overlap.
        era5_futures = _submit_era5(era5_to_fetch, request, era5, root_uri, writer)
        nsrdb_futures = (
            _submit_nsrdb(nsrdb_to_fetch, nreq, nsrdb, root_uri, writer)
            if nreq is not None
            else []
        )

        era5_fetched, era5_failed = _collect_fetches(
            era5_to_fetch, era5_futures, "ERA5"
        )
        nsrdb_fetched, nsrdb_failed = _collect_fetches(
            nsrdb_to_fetch, nsrdb_futures, "NSRDB"
        )
        # Points held back as dead are failures of this request too -- they belong
        # in the ledger and in the silver decision below.
        era5_failed += era5_held
        nsrdb_failed += nsrdb_held
        era5_rows = era5_reusable + era5_fetched
        nsrdb_rows = nsrdb_reusable + nsrdb_fetched
        _require_rows("ERA5", era5_rows, era5_failed)
        if nreq is not None:
            # A narrowed write is a gap this run recorded, not a success to pass
            # over -- counted with the failures so the silver decision below sees it.
            nsrdb_failed += dead_letter.short_writes(nsrdb_fetched, nreq.end_date)
            _require_rows("NSRDB", nsrdb_rows, nsrdb_failed)

        # A permanently unserviceable point is a gap this run created and already
        # recorded; refusing to build over it would be the all-or-nothing behaviour
        # the per-point guards remove. Only *permanent* failures qualify -- a
        # transient one may succeed next run, and building without it would hide that.
        permanent = [
            entry
            for entry in (*era5_failed, *nsrdb_failed)
            if entry.get("permanent") == "True"
        ]
        build_over_gaps = allow_missing_points or bool(permanent)
        if permanent and not allow_missing_points:
            logger.warning(
                "%d point(s) failed permanently this run; building silver over the "
                "remaining points: %s",
                len(permanent),
                [entry["point"] for entry in permanent],
            )
        logger.info("step 2/2: ERA5 x NSRDB silver join")
        era5_nsrdb_silver(
            request=request,
            root_uri=root_uri,
            writer=writer,
            allow_missing_points=build_over_gaps,
            as_of=as_of,
        )

        update_flow_manifest(
            flow_id=flow_id,
            root_uri=root_uri,
            status=FlowStatus.COMPLETED,
            # Keyed by source: the fetches are this flow's own task runs, so there
            # is no per-source row to carry them. One entry naming which source
            # produced what answers the same questions honestly.
            metadata={
                "output_write_ids": {
                    "era5": [row.write_id for row in era5_rows],
                    "nsrdb": [row.write_id for row in nsrdb_rows],
                },
                "failed_points": {"era5": era5_failed, "nsrdb": nsrdb_failed},
            },
        )
    except Exception as exc:
        update_flow_manifest(
            flow_id=flow_id, root_uri=root_uri, status=FlowStatus.FAILED, error=exc
        )
        raise
    logger.info(
        "finished climate pipeline: ERA5 %d + NSRDB %d bronze write(s) -> silver",
        len(era5_rows),
        len(nsrdb_rows),
    )
