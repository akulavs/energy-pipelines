"""
Prefect flows for the load pipeline.

Every flow takes the geographies
(:class:`~external_data.load_pipeline.schema.LoadGeographies` -- one or more PUMAs)
plus only what that flow needs, so a deployment's run form shows exactly those
schemas:

- :func:`ingest_resstock` -- residential bronze: each PUMA's metadata and its
  per-building timeseries.
- :func:`ingest_comstock` -- commercial bronze, in the same shape.
- :func:`ingest_dsgrid` -- industrial bronze. Only the *states* are used: dsgrid
  publishes per state, so one ingest serves every PUMA in it.
- :func:`industrial_load_silver` -- stack both dsgrid sources into one metadata
  table and one timeseries table per state. States only, as above.
- :func:`run_load_pipeline` -- the three bronze ingests as overlapping tasks, then
  industrial silver as a subflow.

**A run is a list of PUMAs**, and nothing is keyed by the list. Each PUMA is its own
writes, so a PUMA asked for alone and in a group of five resolves to the same
dataset, and an overlapping group pays only for what it adds. ``PUMAS_IN_FLIGHT``
bounds how many are fetched at once -- a memory bound, since a PUMA's whole table is
resident while it is assembled. What meters OEDI is the pair of global limits below,
not the shape of the loop.

**Every bronze write is keyed by one PUMA**, so the manifest is the PUMA index (as
the climate pipeline's is a point index). Keys and source files do not line up
one-to-one, so fetches are grouped by what OEDI publishes: ResStock metadata is one
~54 MB file per *state*, read once above the PUMA loop and cut into a write per
PUMA; dsgrid is per state too.

**A task run per unit of work**, so a run is legible while it runs: one per building
fetched, one per dsgrid source file, one per silver table. ResStock metadata is one
task per *state*, because one download is genuinely the unit.

All data lands under a single ``root_uri``, distinguished by dataset name.
Deployments point it outside the repo; ``DEFAULT_ROOT`` is only a fallback for
ad-hoc calls.

The building-stock ingests are the expensive ones: one file per building, so a real
PUMA is on the order of a thousand requests per source, times the PUMAs asked for. A
location means **all** its buildings, with no cap -- a partial slice would write
under the same key as a complete PUMA. The building count is logged before each
fan-out starts, which is what an operator watches. ``schema.MAX_PUMAS_PER_RUN``
refuses a longer list rather than shortening it, because a flow's timeout is fixed
at import.

Every bronze flow consults the manifest before fetching, which makes a re-run cheap,
a second PUMA in an ingested state nearly free, and a killed run resumable. The
ledger is the authority rather than Prefect's TTL-bound result cache.
``force_refresh`` overrides it, for a release re-published under an unchanged key.

Follows ``batch_jobs/AGENTS.md``: resolve/compute + write inside tasks, a
flow-manifest lifecycle (start -> COMPLETED / FAILED); the three bronze flows record
no ``input_ids``, since they fetch from the public OEDI lake.
"""

from __future__ import annotations

import datetime
import logging
import os
import pathlib
import uuid
from collections.abc import Callable, Mapping, Sequence

import httpx
import polars as pl
import prefect
from prefect.concurrency.sync import concurrency, rate_limit
from prefect.task_runners import ThreadPoolTaskRunner

import common.exceptions
from common.storage.flow_manifest import (
    FlowStatus,
    update_flow_manifest,
    write_flow_manifest_start,
)
from common.storage.manifest import ManifestRow, resolve_manifest
from external_data.load_pipeline import coverage, failures
from external_data.load_pipeline import oedi_building_stock as oedi
from external_data.load_pipeline import schema, silver
from external_data.load_pipeline.comstock import bronze as comstock_bronze
from external_data.load_pipeline.dsgrid import bronze as dsgrid_bronze
from external_data.load_pipeline.resstock import bronze as resstock_bronze

# Relative fallback only; deployments override this with an external root_uri.
DEFAULT_ROOT = "load_pipeline"
DEFAULT_WRITER = "load_pipeline"

# The per-building fan-out's budget lives on the Prefect server as two global
# limits, not in this file, so they can be retuned without a deploy. Create them
# once per environment:
#
#     prefect global-concurrency-limit create oedi-api  --limit 28
#     prefect global-concurrency-limit create oedi-rate --limit 600 \
#         --slot-decay-per-second 28
#
# Two limits, because they bound different things: ``oedi-api`` caps how many
# requests are **in flight**, ``oedi-rate`` how fast they **start**. Without the
# second, 20 fast responses immediately start 20 more.
#
# Both figures are self-imposed; OEDI publishes no rate limit. What bounds this
# pipeline is **bandwidth**. A ResStock building file is ~6.3 MB, so 28 in flight
# already draws ~15.7 MB/s (~126 Mbit/s). Measured: 48 in flight doubled mean fetch
# latency (10.8 s -> 18.2 s) for 9% more throughput, the extra requests only queueing
# at the link.
#
# The bucket itself never minded -- 2,224 fetches at 28 concurrent, and a run at 56,
# produced zero retries, against the 5,500 GET/s per prefix S3 documents. So this
# ceiling protects the run's own latency, not the provider. The 600-slot pool lets a
# PUMA's buildings burst, then settles to the decay rate.
#
# 429s, honoured via ``Retry-After``, are the only real feedback. A limit that does
# not exist is a no-op, which is what keeps the offline suite server-free.
OEDI_LIMIT = "oedi-api"
OEDI_RATE_LIMIT = "oedi-rate"


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    """
    A concurrency knob from the environment, or its default.

    Read at **import**, not per run: both knobs feed the task-runner size in a flow
    decorator, so a value arriving with a run is too late. Same seam as
    ``LOAD_PIPELINE_ROOT`` in ``deploy.py`` -- set it where the process starts.

    Raises:
        RuntimeError: If the value is not an integer at or above *minimum*. A typo'd
            setting must not fall back to the default: the run would look healthy and
            be sized for the wrong machine.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        msg = f"{name} must be an integer, got {raw!r}"
        raise RuntimeError(msg) from exc
    if value < minimum:
        msg = f"{name} must be >= {minimum}, got {value}"
        raise RuntimeError(msg)
    return value


# The in-flight budget the server-side limits are set to. Not read from the server (a
# global limit is deliberately retunable without a deploy), but the task pool has to
# be sized against it, so the expected number lives here too.
#
# **Raising the server limit alone does nothing.** A mapped child needs a worker
# before it can ask for an HTTP slot, so the smaller of the two wins: ``oedi-api`` at
# 48 against this at 28 left the fan-out pinned at 28. Move both together.
#
# The default is this machine's measured bandwidth ceiling, not a universal one. A
# host with a fatter pipe -- a VM in the bucket's region -- wants a much larger
# number, and can set one without a code change.
OEDI_IN_FLIGHT = _env_int("LOAD_PIPELINE_OEDI_IN_FLIGHT", 28)

# What a metadata download weighs against the budget above, which is denominated in
# building files (~6.3 MB each).
#
# The metadata fetches are submitted together, so something has to bound them, and
# the honest bound is the same bandwidth budget every other request answers to. A
# ResStock state file is ~34-54 MB, so it occupies the slots it actually costs:
# unweighted, a 20-PUMA run could put ~1 GB in flight at once. ComStock's is one
# PUMA's pre-aggregated rows, small enough to weigh the same as a building.
#
# The effect: concurrent metadata self-limits to roughly three state files, which is
# where the link stops being the constraint anyway.
#
# **This puts a floor under the server-side limit.** ``oedi-api`` has to be at least
# ``RESSTOCK_METADATA_SLOTS``, because a request for more slots than the limit holds
# is never satisfiable: the residential metadata would not run slowly, it would wait
# out the task timeout having never started. The documented 28 leaves ample room, but
# the limit is retunable without a deploy, so an operator lowering it below 8 is
# lowering it past the point where the run works at all.
RESSTOCK_METADATA_SLOTS = 8
COMSTOCK_METADATA_SLOTS = 1

# --------------------------------------------------------------------------- #
# The task pool
# --------------------------------------------------------------------------- #

# How many PUMAs are fetched at once. Not a politeness control: the HTTP budget above
# meters OEDI, and one PUMA's fan-out can saturate it alone. Overlapping PUMAs only
# keeps that budget busy while one of them is parsing, validating or writing.
#
# **The default is one, and it is a memory bound, not a policy.** A PUMA is one
# write, so every building's frame is held until the table can be concatenated:
# ~1.7 GB resident for a PUMA's two sources, roughly double at the concat. Measured,
# per PUMA in flight:
#
#     PUMAs in flight   resident   peak      verdict on 26 GB
#     1                  1.7 GB     3.3 GB   fits
#     2                  3.3 GB     6.7 GB   fits
#     3                  5.0 GB    10.0 GB   fits
#     4                  6.7 GB    13.3 GB   tight
#
# On a bandwidth-bound host, raising it buys almost nothing and risks a lot: the link
# sits idle 2.4 s out of a 720 s fetch window (0.33%), which is the entire prize,
# while an OOM costs a whole PUMA's fetches -- one write has no partial state to
# resume from. Where the link is not the ceiling, the arithmetic flips and this
# becomes the knob that matters. Size it from the host's memory using the table
# above, not from its core count.
PUMAS_IN_FLIGHT = _env_int("LOAD_PIPELINE_PUMAS_IN_FLIGHT", 1)

# Sources submitted per PUMA (ResStock timeseries, ComStock, dsgrid).
SOURCES_PER_PUMA = 3

# Child tasks a parent submits and then blocks on, per PUMA: one per dsgrid source
# file. They come out of the same pool while their parent holds a worker, so they
# must be counted -- otherwise splitting a task for visibility re-creates the
# starvation TASK_WORKERS exists to prevent.
DSGRID_SOURCE_TASKS = len(dsgrid_bronze.DsgridSource)
CHILD_TASKS_PER_PUMA = DSGRID_SOURCE_TASKS

# The metadata fetches are deliberately *not* counted into the pool. They are phases
# at the *flow* level rather than children of a bronze task, so nothing blocks on
# them holding a worker -- the structural reason to run a phase rather than nest a
# task. Being submitted rather than walked, they do take workers, but only during the
# opening phase, before any bronze parent exists to starve, and they map no children,
# so a queued one cannot be waiting on a worker the pool has already given away. What
# bounds the downloads is the HTTP budget and their weights on it, not the worker
# count; sizing the pool for the worst case would nearly double it to no effect.

# **The pool must never be the throttle.** A parent fetch task holds a worker for its
# whole life, including while it blocks on the per-building children it maps -- and
# those children come out of this same pool. A pool sized to the parents alone
# starves the fan-out to nothing: at ``max_workers=3`` with three parents alive, 115
# buildings were fetched strictly one at a time and the HTTP budget bound nothing.
#
# Sized as every parent that may be in flight plus the whole HTTP budget, so the
# fan-out reaches its ceiling with the parents parked, plus slack for the flow's own
# bookkeeping tasks.
TASK_WORKERS = (
    PUMAS_IN_FLIGHT * (SOURCES_PER_PUMA + CHILD_TASKS_PER_PUMA) + OEDI_IN_FLIGHT + 4
)

# --------------------------------------------------------------------------- #
# Timeouts
# --------------------------------------------------------------------------- #

# Derived from one place, so a parent can never sit *below* the sum of its children
# and cancel a run whose children are each still inside their own budget.
#
# The rate is measured, not guessed: against the live OEDI bucket, 8-way concurrent
# per-building fetches sustained ~22 requests/second at a mean file size of 1.85 MB.
# Rounded down, and treated as optimistic -- warm CDN, fast link, and DC, which has a
# single county.
OEDI_REQUESTS_PER_SECOND = 20.0

# One PUMA's per-building fan-out, per the two source READMEs.
RESSTOCK_REQUESTS_PER_PUMA = 200
COMSTOCK_REQUESTS_PER_PUMA = 950

# Pure fetch time at that rate: ~10 s and ~48 s. The budgets below sit two orders of
# magnitude above that deliberately: a timeout catches a hang, it does not hold a job
# to its best case. What the request count misses:
#   - tail latency: one of ten serial fetches took 4.8 s against a 0.24 s median
#   - bandwidth: ComStock is ~1.8 GB per PUMA, on a link that may be far slower
#   - the 33M-row concat and schema validation that follow the fetches
RESSTOCK_FETCH_SECONDS = RESSTOCK_REQUESTS_PER_PUMA / OEDI_REQUESTS_PER_SECOND
COMSTOCK_FETCH_SECONDS = COMSTOCK_REQUESTS_PER_PUMA / OEDI_REQUESTS_PER_SECOND

# Budgets for **one unit of work** -- one PUMA, or one state -- enforced on the
# tasks. That is the level a hang is still catchable at: a flow budget wide enough
# for twenty PUMAs would let one stuck PUMA sit for a day.
BUILDING_STOCK_TIMEOUT_SECONDS = 3600
# A metadata step is one download and a projection -- ResStock's ~54 MB state file,
# or ComStock's per-PUMA file -- not an hour of per-building fetching. Its own
# constant because it is also its own *phase*, run before the PUMA loop.
METADATA_TIMEOUT_SECONDS = 1800
# dsgrid is two downloads (~15 MB + ~2.6 MB) plus a local HDF5 reconstruction.
DSGRID_TIMEOUT_SECONDS = 1800
# Silver is local only: read four bronze tables, stack, write (237k rows for DC).
SILVER_TIMEOUT_SECONDS = 1800

# The flow budgets are then the per-unit budgets times the most units a run may
# hold, because a flow's timeout is fixed when this module is imported and cannot
# be widened for the run in front of it. Each is a ceiling on a whole run rather
# than a statement about a typical one; the per-task budgets above are what
# actually catch a hang.
MAX_UNITS_PER_RUN = schema.MAX_PUMAS_PER_RUN

# The metadata is a **phase of its own**, ahead of the PUMA loop, so it is a separate
# term rather than something the per-PUMA budget absorbs. Its fetches overlap, but
# the budget assumes they do not: at worst a run's PUMAs are in as many states as
# there are PUMAs, so the phase is bounded by the same unit count.
METADATA_PHASE_TIMEOUT_SECONDS = METADATA_TIMEOUT_SECONDS * MAX_UNITS_PER_RUN

BUILDING_STOCK_FLOW_TIMEOUT_SECONDS = (
    METADATA_PHASE_TIMEOUT_SECONDS + BUILDING_STOCK_TIMEOUT_SECONDS * MAX_UNITS_PER_RUN
)
DSGRID_FLOW_TIMEOUT_SECONDS = DSGRID_TIMEOUT_SECONDS * MAX_UNITS_PER_RUN
SILVER_FLOW_TIMEOUT_SECONDS = SILVER_TIMEOUT_SECONDS * MAX_UNITS_PER_RUN
# Within one PUMA the three bronze fetches run **concurrently**, so the parent needs
# the slowest of them rather than their sum -- once per PUMA, plus a silver build per
# state (at most one per PUMA), plus the metadata phase ahead of all of it.
PIPELINE_TIMEOUT_SECONDS = METADATA_PHASE_TIMEOUT_SECONDS + MAX_UNITS_PER_RUN * (
    max(BUILDING_STOCK_TIMEOUT_SECONDS, DSGRID_TIMEOUT_SECONDS) + SILVER_TIMEOUT_SECONDS
)


def _resolve_root(root_uri: str) -> str:
    """A relative root_uri (e.g. "load_pipeline") has no scheme; resolve it to an
    absolute path so the manifest/parquet writers accept it."""
    if "://" not in str(root_uri):
        return str(pathlib.Path(root_uri).resolve())
    return root_uri


def _run_geographies(geographies: schema.LoadGeographies) -> list[dict[str, str]]:
    """
    The geographies a run was for, for its flow manifest -- one entry per PUMA.

    The dataset keys already carry the PUMA; what they cannot say is which PUMAs were
    asked for *together*, which is what this records -- the run, as opposed to what
    the run produced. A list rather than a joined string, so a reader can iterate it.
    """
    return [
        {
            "puma_gisjoin": geography.puma_gisjoin,
            "state": geography.state,
            "utc_offset_minutes": str(geography.utc_offset_minutes),
        }
        for geography in geographies.pumas
    ]


def _report_completeness(
    dataset_name: str,
    puma_gisjoin: str,
    expected: int,
    unavailable: int,
    newly_failed: int,
) -> None:
    """
    Say what the PUMA's write adds up to, and warn only when a re-run would help.

    Derived from three numbers the flow already holds -- every building the metadata
    lists, minus those the release has already refused, minus those it refused this
    run -- so it needs no manifest field and no data read.

    Three rather than two, because two cannot tell an operator what to do: a PUMA
    short by exactly ``unavailable`` is as complete as it will ever get, and advising
    a re-run would spend a thousand requests on files that refuse again. Only a gap
    beyond those is worth re-running.
    """
    logger = prefect.get_run_logger()
    serviceable = expected - unavailable
    written = serviceable - newly_failed
    if written >= serviceable:
        if unavailable:
            logger.info(
                "%s for PUMA %s: %d of %d building(s); the other %d are recorded "
                "unavailable, so this PUMA is as complete as the release allows. "
                "Sums and peaks read off it understate demand by that share.",
                dataset_name,
                puma_gisjoin,
                written,
                expected,
                unavailable,
            )
        else:
            logger.info(
                "%s for PUMA %s: %d building(s), complete",
                dataset_name,
                puma_gisjoin,
                written,
            )
        return
    logger.warning(
        "%s for PUMA %s: %d of %d building(s) -- %d short beyond the %d recorded "
        "unavailable, newly refused this run. A re-run will ask for them again "
        "only if they are transient; the permanent ones are now on this run's "
        "manifest. Aggregates read now understate demand.",
        dataset_name,
        puma_gisjoin,
        written,
        expected,
        newly_failed,
        unavailable,
    )


def _should_retry(task: object, task_run: object, state: prefect.State) -> bool:
    """
    Retry unless trying again cannot help.

    A permanently absent building file burns both retries for nothing. Transient is
    the default: a needlessly retried transient costs seconds, while treating a
    transient as permanent drops a building from the write.
    """
    try:
        exc = state.result(raise_on_failure=False)
    except Exception:  # noqa: BLE001 - a state we cannot read is not classifiable
        return True
    return not failures.is_permanent(exc if isinstance(exc, BaseException) else None)


def _surviving(
    geographies: schema.LoadGeographies,
    rejected: Sequence[Mapping[str, str]],
    source: str,
) -> tuple[schema.LoadGeography, ...]:
    """
    The PUMAs *source* can still be asked about, in the order given.

    Per source, not pooled: ResStock and ComStock come from separate releases, so a
    code missing from one is no evidence about the other. Pooling would let one
    release's gap silently drop data the other publishes.
    """
    dropped = failures.rejected_pumas(rejected, source)
    return tuple(g for g in geographies.pumas if g.puma_gisjoin not in dropped)


def _still_serviceable(
    geographies: schema.LoadGeographies,
    rejected: Sequence[Mapping[str, str]],
) -> tuple[schema.LoadGeography, ...]:
    """
    The PUMAs at least one building-stock source can still serve.

    What the *state*-grained steps key off -- dsgrid bronze and the industrial
    silver. Both cover a whole state whatever PUMA named it, so a state keeps its
    place as long as one of its PUMAs was real. A state whose every PUMA was a bad
    code is dropped: nothing in the run actually asked for that state, a typo
    reached it, and ingesting it would answer a question nobody put.
    """
    live = {g.puma_gisjoin for g in _surviving(geographies, rejected, "resstock")}
    live |= {g.puma_gisjoin for g in _surviving(geographies, rejected, "comstock")}
    return tuple(g for g in geographies.pumas if g.puma_gisjoin in live)


def _report_rejected(
    rejected: Sequence[Mapping[str, str]],
    total: int,
    serviceable: Sequence[schema.LoadGeography],
) -> None:
    """
    Say which codes were dropped and what is still being run.

    Split by what the rejection actually cost, because they are different news. A
    code no source in the run can serve is a PUMA that will not be produced --
    nearly always a typo, and the operator's to fix. A code only one release lacks
    is still ingested from the other, and calling that "skipped" would send someone
    hunting a mistake in a GISJOIN that is perfectly good. *serviceable* is what
    survived, so it is what separates the two.
    """
    if not rejected:
        return
    logger = prefect.get_run_logger()
    still_run = {g.puma_gisjoin for g in serviceable}
    dropped = [e for e in rejected if e.get("puma_gisjoin", "?") not in still_run]
    partial = [e for e in rejected if e.get("puma_gisjoin", "?") in still_run]
    if dropped:
        codes = {entry.get("puma_gisjoin", "?") for entry in dropped}
        logger.warning(
            "%d of %d requested PUMA(s) are not in the release and were skipped -- "
            "%s. Check the GISJOIN codes (they are 2010-census vintage). The run "
            "continues for the other %d; the skipped codes are on this run's flow "
            "manifest under %r.",
            len(codes),
            total,
            failures.describe_rejected(dropped),
            total - len(codes),
            failures.REJECTED_PUMAS_KEY,
        )
    if partial:
        codes = {entry.get("puma_gisjoin", "?") for entry in partial}
        logger.warning(
            "%d requested PUMA(s) are missing from one release but published by "
            "the other, and are ingested from that source alone -- %s. Not a bad "
            "code: the releases are built separately, so a real PUMA can be in one "
            "and not the other. Recorded on this run's flow manifest under %r.",
            len(codes),
            failures.describe_rejected(partial),
            failures.REJECTED_PUMAS_KEY,
        )


def _refuse_if_nothing_left(
    surviving: Sequence[schema.LoadGeography],
    rejected: Sequence[Mapping[str, str]],
) -> None:
    """
    Fail the run when every requested PUMA was a bad code.

    Tolerance is for *finishing the work that exists* -- with none left there is no
    run to save, and completing would report success for a run that wrote nothing.
    The same rule the climate pipeline applies to a gap nothing explains.
    """
    if surviving:
        return
    msg = (
        "every requested PUMA was rejected by the release, so there is nothing to "
        f"ingest -- {failures.describe_rejected(rejected)}. Check the GISJOIN "
        "codes: they are 2010-census vintage, and a PUMA split since then has "
        "sub-codes rather than the parent."
    )
    raise common.exceptions.PipelineValueError(msg)


# Used only on the interruption path, where the Prefect run logger may no longer
# be usable: the process is on its way out and its run context may already be gone.
_logger = logging.getLogger(__name__)


def _record_interruption(
    flow_id: str,
    root_uri: str,
    learned: Mapping[str, list[dict[str, str]]],
) -> None:
    """
    Close out a run stopped from outside, keeping what it had already discovered.

    Cancelling a flow run sends SIGTERM, which Prefect turns into a
    ``TerminationSignal`` -- a ``BaseException``, so the ``except Exception`` that
    records a failure never sees it. The manifest is then left saying ``running``
    forever, with the run's records lost in memory. Same loss the failure path
    avoids; it just needs a ``finally`` to catch the exits an ``except`` cannot.

    Errors here are swallowed deliberately: this runs while a termination signal is
    propagating, and that signal must reach the engine. A manifest we could not write
    is worth less than an interruption we turned into a hang.
    """
    try:
        update_flow_manifest(
            flow_id=flow_id,
            root_uri=root_uri,
            status=FlowStatus.CANCELLED,
            metadata=dict(learned),
        )
    except Exception:  # noqa: BLE001 - see above; the signal must win
        _logger.warning(
            "could not record the interrupted run %s; its manifest stays 'running'",
            flow_id,
            exc_info=True,
        )


def _flow_prelude() -> tuple[datetime.datetime, datetime.datetime, str]:
    """Common opening values: (run_ts, scheduled_time, flow_id)."""
    run_ts = datetime.datetime.now(tz=datetime.UTC)
    scheduled_time = prefect.runtime.flow_run.scheduled_start_time or run_ts
    flow_id = prefect.runtime.flow_run.id or str(uuid.uuid4())
    return run_ts, scheduled_time, flow_id


def _ingest_resstock_metadata(
    geographies: schema.LoadGeographies,
    resstock: schema.ResstockRequestArgs,
    root_uri: str,
    writer: str,
    force_refresh: bool,
    rejected: list[dict[str, str]],
) -> list[ManifestRow]:
    """
    The residential metadata for every PUMA of the run, a state file at a time but
    all of them in flight together.

    Ahead of the timeseries rather than beside it: the fetches read these writes for
    their building ids, and a metadata write landing first is what keeps a profile
    from ever being visible without its weights.

    Also the run's first sight of a PUMA the release has no such code for.
    ``rejected`` is owned by the caller and appended to, like ``failed`` on the
    timeseries side, so a run that dies later has still recorded which codes were
    bad.

    A missing *state* file is deliberately still fatal. A well-formed GISJOIN always
    maps to one of the 51 states these releases publish (``state_code_for_fips``
    refuses the rest), so a 404 on the state object means the release name is wrong --
    which is wrong for every state equally, and not something to half-finish a run
    over. Only the per-PUMA verdict is tolerated here.
    """
    logger = prefect.get_run_logger()
    by_state = geographies.by_state()
    logger.info(
        "residential metadata: %d state file(s) together -- %s",
        len(by_state),
        ", ".join(f"{state} ({len(pumas)} PUMA)" for state, pumas in by_state.items()),
    )
    # Submitted together, not walked. The states are independent downloads, so
    # walking them made the run's opening a queue of round-trips that nothing else
    # could start behind. The HTTP budget bounds how many actually fly at once.
    futures = [
        fetch_resstock_metadata_bronze.submit(
            pumas, resstock, root_uri, writer, force_refresh
        )
        for pumas in by_state.values()
    ]
    rows: list[ManifestRow] = []
    for future in futures:
        state_rows, state_rejected = future.result()
        rows.extend(state_rows)
        rejected.extend(state_rejected)
    return rows


def _ingest_comstock_metadata(
    geographies: schema.LoadGeographies,
    comstock: schema.ComstockRequestArgs,
    root_uri: str,
    writer: str,
    force_refresh: bool,
    rejected: list[dict[str, str]],
) -> list[ManifestRow]:
    """
    The commercial metadata for every PUMA of the run, all in flight together.

    ComStock publishes per PUMA, so the unit here is the PUMA where the residential
    counterpart's is the state. What they share is the shape: a phase of its own, at
    the flow level, ahead of the timeseries.

    A phase rather than a task nested inside ``fetch_comstock_bronze``, which would
    hold a runner worker while blocking on its own child. It also makes the ordering
    structural -- every metadata write is older than every timeseries write, so a
    point-in-time read cannot find profiles without their weights.
    """
    logger = prefect.get_run_logger()
    logger.info(
        "commercial metadata: %d PUMA file(s) together -- %s",
        len(geographies.pumas),
        ", ".join(g.puma_gisjoin for g in geographies.pumas),
    )
    # One small request each and no dependency between them, so they go together:
    # walked, a twenty-PUMA run spent twenty round-trips before any fetch started.
    submitted = [
        (
            geography,
            fetch_comstock_metadata_bronze.submit(
                geography, comstock, root_uri, writer, force_refresh
            ),
        )
        for geography in geographies.pumas
    ]
    rows: list[ManifestRow] = []
    for geography, future in submitted:
        try:
            rows.append(future.result())
        except failures.PermanentFetchError as exc:
            # ComStock publishes metadata per PUMA, so a code it has no file for
            # surfaces as a 404 -- already classified permanent at the raise site.
            # Same verdict as the residential side reaches by reading the state
            # file: record the code and keep the rest of the list.
            logger.warning(
                "no ComStock metadata for PUMA %s; skipping it and continuing "
                "with the rest of the request (%s)",
                geography.puma_gisjoin,
                exc,
            )
            rejected.append(
                failures.rejected(geography.puma_gisjoin, "comstock", str(exc))
            )
    return rows


def _one_per_state(
    pumas: Sequence[schema.LoadGeography],
) -> tuple[schema.LoadGeography, ...]:
    """
    The first of *pumas* given for each distinct state.

    The list-shaped counterpart of ``LoadGeographies.one_per_state``, taken here
    because the state-grained steps now run over the PUMAs that survived rather than
    over everything the run asked for.
    """
    first: dict[str, schema.LoadGeography] = {}
    for geography in pumas:
        first.setdefault(geography.state, geography)
    return tuple(first.values())


def _ingest_each_unit(
    task: prefect.Task,
    pumas: Sequence[schema.LoadGeography],
    args: schema.ResstockRequestArgs
    | schema.ComstockRequestArgs
    | schema.DsgridRequestArgs,
    root_uri: str,
    writer: str,
    force_refresh: bool,
    failed: list[dict[str, str]],
    *,
    by_state: bool = False,
) -> list[ManifestRow]:
    """
    Run one source's fetch task over a run's geographies, and pool what they wrote.

    Takes the PUMAs still worth fetching rather than the run's whole request: a code
    the metadata phase found the release has no file for is already recorded, and
    submitting its timeseries would only fail again, slower.

    The unit is the PUMA, except for state-published sources (``by_state``), where one
    bronze serves every PUMA in a state.

    ``failed`` is owned by the caller and appended to as each unit reports, rather
    than returned at the end: a run that dies partway has still discovered which
    buildings the release cannot serve, and returning them would discard that on
    exactly the runs that most need it kept.

    Units are **submitted** ``PUMAS_IN_FLIGHT`` at a time, so their fan-outs overlap
    and the HTTP budget stays busy while any one of them is parsing or writing.
    Results are collected in submission order, so ``failed`` reads back PUMA by PUMA.
    A window rather than the whole list, because what bounds this is memory, not OEDI.
    """
    logger = prefect.get_run_logger()
    units = _one_per_state(pumas) if by_state else tuple(pumas)
    if by_state and len(units) < len(pumas):
        logger.info(
            "%d PUMA(s) span %d state(s); this source is published per state, so it "
            "is ingested once each",
            len(pumas),
            len(units),
        )

    rows: list[ManifestRow] = []
    noun = "state" if by_state else "PUMA"
    for start in range(0, len(units), PUMAS_IN_FLIGHT):
        window = units[start : start + PUMAS_IN_FLIGHT]
        logger.info(
            "%s %d-%d of %d: %s",
            noun,
            start + 1,
            start + len(window),
            len(units),
            ", ".join(g.state if by_state else g.puma_gisjoin for g in window),
        )
        futures = [
            task.submit(geography, args, root_uri, writer, force_refresh)
            for geography in window
        ]
        for future in futures:
            unit_rows, unit_failed = future.result()
            rows.extend(unit_rows)
            failed.extend(unit_failed)
    return rows


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #


@prefect.task(
    retries=2,
    retry_delay_seconds=[5, 20],
    retry_jitter_factor=0.5,
    # A building the release cannot serve should not burn its retries first.
    retry_condition_fn=_should_retry,
)
def fetch_resstock_building(
    bldg_id: int, args: resstock_bronze.ResstockPumaTimeseriesRequestArgs
) -> pl.DataFrame:
    """
    One ResStock building's timeseries -- the unit of concurrency and of retry.

    Per building rather than per PUMA, so an hour-long ingest shows progress and one
    bad building can be retried without redoing the rest; the jitter keeps a whole
    fan-out from retrying in lockstep after a blip.

    Both limits are acquired **inside** the body, so a retry re-acquires them rather
    than bypassing the budget. A client per building rather than a shared pool: a
    task's arguments must be serializable, so the pool cannot be passed in -- the
    cost is a handshake per building, which the concurrency hides.
    """
    with concurrency(OEDI_LIMIT, occupy=1):
        rate_limit(OEDI_RATE_LIMIT)
        with httpx.Client(timeout=120, follow_redirects=True) as client:
            return resstock_bronze.fetch_building_frame(args, bldg_id, client)


@prefect.task(
    retries=2,
    retry_delay_seconds=[5, 20],
    retry_jitter_factor=0.5,
    # A building the release cannot serve should not burn its retries first.
    retry_condition_fn=_should_retry,
)
def fetch_comstock_building(
    bldg_id: int, args: comstock_bronze.ComstockPumaTimeseriesRequestArgs
) -> pl.DataFrame:
    """One ComStock building's timeseries; see :func:`fetch_resstock_building`."""
    with concurrency(OEDI_LIMIT, occupy=1):
        rate_limit(OEDI_RATE_LIMIT)
        with httpx.Client(timeout=120, follow_redirects=True) as client:
            return comstock_bronze.fetch_building_frame(args, bldg_id, client)


def _mapped_building_fetcher(
    task: prefect.Task,
    args: resstock_bronze.ResstockPumaTimeseriesRequestArgs
    | comstock_bronze.ComstockPumaTimeseriesRequestArgs,
    source: str,
    failed: list[dict[str, str]],
) -> oedi.BuildingFetcher:
    """
    A fetcher that maps the per-building task over a whole PUMA's ids.

    Injected into the package's ingest, which owns reuse and the write: the
    concurrency lives here and the domain logic stays there, with no Prefect import
    on that side.

    Every building is mapped in one call. The fan-out is bounded by the two global
    limits the task body acquires rather than by the length of the list, so mapping
    wide is safe -- what it costs is memory, since every frame is held until the
    concat.

    Collects **states**, not results, so one absent building cannot discard the
    hundreds that succeeded alongside it:

    - *Permanent* (the release cannot serve this file): drop the building, record it,
      and let the PUMA be written short. The record is what stops a later run asking
      again, and what ``_report_completeness`` reports against.
    - *Transient*: re-raise, so the whole PUMA is retried. That data is expected to
      arrive, and writing a short PUMA over it would make a blip indistinguishable
      from a permanent hole, which no later run would look at again.
    """

    def _fetch(
        bldg_ids: Sequence[int],
    ) -> tuple[pl.DataFrame | None, list[int]]:
        ordered = list(bldg_ids)
        futures = task.map(ordered, prefect.unmapped(args))
        frames, fetched = [], []
        for bldg_id, future in zip(ordered, futures, strict=True):
            future.wait()
            state = future.state
            if state.is_completed():
                frames.append(state.result())
                fetched.append(bldg_id)
                continue
            exc = state.result(raise_on_failure=False)
            if not failures.is_permanent(
                exc if isinstance(exc, BaseException) else None
            ):
                if isinstance(exc, BaseException):
                    raise exc
                msg = f"building {bldg_id} failed: {state.message}"
                raise RuntimeError(msg)
            failed.append(
                failures.record(
                    bldg_id,
                    args.puma_gisjoin,
                    source,
                    permanent=True,
                    error=str(exc),
                )
            )
        if not fetched:
            return None, []
        return oedi.assemble_timeseries_table(frames), fetched

    return _fetch


@prefect.task(timeout_seconds=METADATA_TIMEOUT_SECONDS)
def fetch_resstock_metadata_bronze(
    geographies: Sequence[schema.LoadGeography],
    resstock: schema.ResstockRequestArgs,
    root_uri: str,
    writer: str,
    force_refresh: bool = False,
) -> tuple[list[ManifestRow], list[dict[str, str]]]:
    """
    The residential metadata bronze for every PUMA of **one state**.

    One write per PUMA, holding that PUMA's buildings and weights, like every other
    dataset here. But OEDI publishes the file per state, so the *fetch* is hoisted:
    a five-PUMA run pulls 54 MB once and cuts five writes from it.

    The ledger is consulted for all of them in one scan, so a run adding a sixth PUMA
    to an ingested state downloads once and writes once, and a re-run of PUMAs already
    on disk downloads nothing.

    Returns ``(rows, rejected)``: the writes, and a record per PUMA this state's file
    has no such code for. The hoisted download is exactly why the rejection cannot be
    allowed to raise -- the other PUMAs of the state are already parsed and in memory.
    """
    logger = prefect.get_run_logger()
    params = [
        silver.resstock_metadata_request(geography, resstock)
        for geography in geographies
    ]
    to_write, covered = coverage.split_by_coverage(
        resstock_bronze.METADATA_DATASET_NAME,
        params,
        root_uri,
        force_refresh=force_refresh,
    )
    for row in covered:
        logger.info(
            "reusing %s -> %s", resstock_bronze.METADATA_DATASET_NAME, row.data_uri
        )
    rows = list(covered)
    rejected: list[dict[str, str]] = []
    if to_write:
        # Metered like every other request, and inside the body so a retry
        # re-acquires rather than bypasses the budget -- the same rule the
        # per-building fetches follow. This is what makes it safe to submit the
        # states together instead of walking them.
        with concurrency(OEDI_LIMIT, occupy=RESSTOCK_METADATA_SLOTS):
            rate_limit(OEDI_RATE_LIMIT)
            written, rejected = resstock_bronze.ingest_metadata_bronze_for_pumas(
                to_write,
                root_uri=root_uri,
                writer=writer,
                write_time=datetime.datetime.now(tz=datetime.UTC),
            )
        rows.extend(written)
    return rows, rejected


@prefect.task(timeout_seconds=BUILDING_STOCK_TIMEOUT_SECONDS)
def fetch_resstock_bronze(
    geography: schema.LoadGeography,
    resstock: schema.ResstockRequestArgs,
    root_uri: str,
    writer: str,
    force_refresh: bool = False,
) -> tuple[list[ManifestRow], list[dict[str, str]]]:
    """
    Ingest one PUMA's residential timeseries bronze, against the metadata bronze
    :func:`fetch_resstock_metadata_bronze` has already written for it.

    The metadata is fetched in its own phase because one state file serves a run's
    whole list, which only works above the PUMA loop. Ordering carries the invariant
    the shared ``write_time`` used to: the metadata write always precedes the
    timeseries write, so any read that finds the profiles finds their weights.
    """
    logger = prefect.get_run_logger()
    logger.info(
        "input params -> geography=%s resstock=%s force_refresh=%s",
        geography.model_dump_json(),
        resstock.model_dump_json(),
        force_refresh,
    )
    write_time = datetime.datetime.now(tz=datetime.UTC)
    rows: list[ManifestRow] = []

    # Ids from this PUMA's metadata bronze rather than a second download of the
    # state file (~54 MB for a large state).
    metadata_params = silver.resstock_metadata_request(geography, resstock)
    timeseries_params = silver.resstock_timeseries_request(geography, resstock)
    bldg_ids = resstock_bronze.building_ids(
        resstock_bronze.read_metadata_bronze(metadata_params, root_uri),
        geography.puma_gisjoin,
    )
    # Reuse lives in the ingest, which owns the write key. Read the ledger back
    # before spending anything: buildings earlier runs found permanently unavailable
    # are dropped, which is what lets the PUMA settle instead of re-failing on the
    # same file every run.
    dead = failures.known_dead(root_uri, "resstock", geography.puma_gisjoin)
    if dead:
        logger.info("skipping %d building(s) recorded unavailable", len(dead))
    failed: list[dict[str, str]] = []
    write_rows = resstock_bronze.ingest_puma_timeseries_bronze(
        timeseries_params,
        root_uri=root_uri,
        writer=writer,
        write_time=write_time,
        bldg_ids=bldg_ids,
        force_refresh=force_refresh,
        skip_bldg_ids=dead,
        fetch_buildings=_mapped_building_fetcher(
            fetch_resstock_building, timeseries_params, "resstock", failed
        ),
    )
    _report_completeness(
        resstock_bronze.TIMESERIES_DATASET_NAME,
        geography.puma_gisjoin,
        len(bldg_ids),
        len(dead),
        len(failed),
    )
    rows.extend(write_rows)
    return rows, failed


@prefect.task(timeout_seconds=METADATA_TIMEOUT_SECONDS)
def fetch_comstock_metadata_bronze(
    geography: schema.LoadGeography,
    comstock: schema.ComstockRequestArgs,
    root_uri: str,
    writer: str,
    force_refresh: bool = False,
) -> ManifestRow:
    """
    One PUMA's commercial metadata bronze: the ids to fetch and each building's
    expansion weight, with the census-tract duplication already collapsed.

    Its own task, run from :func:`_ingest_comstock_metadata` as a phase before the
    timeseries, like the residential side. It is one request rather than minutes of
    work, but a task of its own so it has a row an operator can see.

    Keyed per PUMA, like the file it comes from, so the ledger answers for exactly the
    geography asked about -- the one thing the two metadata phases do not share, since
    the residential file is per state.
    """
    logger = prefect.get_run_logger()
    params = silver.comstock_puma_metadata_request(geography, comstock)
    covered = coverage.covered_write(
        comstock_bronze.PUMA_METADATA_DATASET_NAME,
        params,
        root_uri,
        force_refresh=force_refresh,
    )
    if covered is not None:
        logger.info(
            "reusing %s -> %s",
            comstock_bronze.PUMA_METADATA_DATASET_NAME,
            covered.data_uri,
        )
        return covered
    with concurrency(OEDI_LIMIT, occupy=COMSTOCK_METADATA_SLOTS):
        rate_limit(OEDI_RATE_LIMIT)
        return comstock_bronze.ingest_puma_metadata_bronze(
            params,
            root_uri=root_uri,
            writer=writer,
            write_time=datetime.datetime.now(tz=datetime.UTC),
        )


@prefect.task(timeout_seconds=BUILDING_STOCK_TIMEOUT_SECONDS)
def fetch_comstock_bronze(
    geography: schema.LoadGeography,
    comstock: schema.ComstockRequestArgs,
    root_uri: str,
    writer: str,
    force_refresh: bool = False,
) -> tuple[list[ManifestRow], list[dict[str, str]]]:
    """
    Ingest one PUMA's commercial timeseries bronze, against the metadata bronze
    :func:`fetch_comstock_metadata_bronze` has already written for it.

    Mirrors :func:`fetch_resstock_bronze` deliberately -- one row per building of
    metadata, one row per building x timestep of load, joined on ``bldg_id`` -- so the
    two building-stock sources read alike. The metadata is **not** fetched here; its
    phase has already written it, which is what keeps the metadata write the older of
    the two and lets this task stamp its ``write_time`` at the top.
    """
    logger = prefect.get_run_logger()
    logger.info(
        "input params -> geography=%s comstock=%s force_refresh=%s",
        geography.model_dump_json(),
        comstock.model_dump_json(),
        force_refresh,
    )
    write_time = datetime.datetime.now(tz=datetime.UTC)
    rows: list[ManifestRow] = []
    metadata_params = silver.comstock_puma_metadata_request(geography, comstock)

    # Ids from the metadata phase's write rather than a second download of the PUMA
    # file.
    bldg_ids = comstock_bronze.building_ids(
        comstock_bronze.read_puma_metadata_bronze(metadata_params, root_uri)
    )
    dead = failures.known_dead(root_uri, "comstock", geography.puma_gisjoin)
    if dead:
        logger.info("skipping %d building(s) recorded unavailable", len(dead))
    failed: list[dict[str, str]] = []
    timeseries_params = silver.comstock_timeseries_request(geography, comstock)
    write_rows = comstock_bronze.ingest_puma_timeseries_bronze(
        timeseries_params,
        root_uri=root_uri,
        writer=writer,
        write_time=write_time,
        bldg_ids=bldg_ids,
        force_refresh=force_refresh,
        skip_bldg_ids=dead,
        fetch_buildings=_mapped_building_fetcher(
            fetch_comstock_building, timeseries_params, "comstock", failed
        ),
    )
    _report_completeness(
        comstock_bronze.TIMESERIES_DATASET_NAME,
        geography.puma_gisjoin,
        len(bldg_ids),
        len(dead),
        len(failed),
    )
    rows.extend(write_rows)
    return rows, failed


@prefect.task(timeout_seconds=DSGRID_TIMEOUT_SECONDS)
def fetch_dsgrid_source_bronze(
    params: dsgrid_bronze.DsgridRequestArgs,
    root_uri: str,
    writer: str,
    force_refresh: bool = False,
) -> list[ManifestRow]:
    """
    One dsgrid source file for one state -- the unit of work and of visibility.

    A source is one download of a national ``.dsg``, one HDF5 reconstruction, and two
    dataset writes. One task per file rather than one for both: an operator can then
    tell a slow ``industrial.dsg`` from a stalled one, and the two downloads overlap.

    A source is skipped only when BOTH of its datasets are present. Both come from the
    same parse, so a half-written source is redone whole -- harmless, since every
    write is its own immutable version.
    """
    logger = prefect.get_run_logger()
    profile = dsgrid_bronze.PROFILES[params.source]
    present = [
        coverage.covered_write(name, params, root_uri, force_refresh=force_refresh)
        for name in (profile.metadata_dataset_name, profile.timeseries_dataset_name)
    ]
    if all(row is not None for row in present):
        logger.info("reusing dsgrid %s bronze (both tables present)", params.source)
        return [row for row in present if row is not None]

    # The download + reconstruct lives in the package -- batch_jobs holds no domain
    # logic. The key is passed in rather than rebuilt from state + source, which
    # would take the *default* for every other field: an override would then write
    # under one key and resolve under another.
    rows = dsgrid_bronze.ingest_all(
        requests=[params],
        root_uri=root_uri,
        writer=writer,
        write_time=datetime.datetime.now(tz=datetime.UTC),
    )
    for row in rows:
        logger.info("dsgrid bronze %s -> %s", row.dataset_name, row.data_uri)
    return rows


@prefect.task(timeout_seconds=DSGRID_TIMEOUT_SECONDS)
def fetch_dsgrid_bronze(
    geography: schema.LoadGeography,
    dsgrid: schema.DsgridRequestArgs,
    root_uri: str,
    writer: str,
    force_refresh: bool = False,
) -> tuple[list[ManifestRow], list[dict[str, str]]]:
    """
    Ingest the industrial bronze for the state: the requested dsgrid source
    file(s), which together make up the industrial total.

    dsgrid is published per state and covers every county at once, so only
    ``geography.state`` is used and the same bronze serves every PUMA in it. Maps
    :func:`fetch_dsgrid_source_bronze` over the requested files, so each is its own
    task run and the reuse decision is made per source -- the grain it belongs at.
    """
    logger = prefect.get_run_logger()
    logger.info(
        "input params -> geography=%s dsgrid=%s force_refresh=%s",
        geography.model_dump_json(),
        dsgrid.model_dump_json(),
        force_refresh,
    )
    # Build the bronze keys in ONE place -- the same helper silver resolves with --
    # so every field of the request reaches the manifest.
    requests = silver.dsgrid_requests(geography.state, dsgrid)
    logger.info(
        "state %s: %d dsgrid source file(s) -- %s",
        geography.state,
        len(requests),
        ", ".join(str(params.source) for params in requests),
    )
    futures = fetch_dsgrid_source_bronze.map(
        requests,
        prefect.unmapped(root_uri),
        prefect.unmapped(writer),
        prefect.unmapped(force_refresh),
    )
    rows: list[ManifestRow] = []
    for future in futures:
        rows.extend(future.result())
    # The same shape as the building-stock tasks, with nothing to report: dsgrid is
    # whole-file downloads, so there is no per-unit failure to tolerate.
    return rows, []


@prefect.task()
def resolve_industrial_inputs(
    request: schema.IndustrialLoadRequestArgs,
    root_uri: str,
    as_of: datetime.datetime | None = None,
) -> dict[str, ManifestRow]:
    """
    Resolve the bronze pair each silver table stacks, for lineage.

    Metadata and timeseries per requested source -- four datasets when both sources
    are requested. ``as_of`` bounds the resolve so the recorded ``write_id``s match
    the data the build reads.
    """
    logger = prefect.get_run_logger()
    logger.info(
        "resolving dsgrid bronze under %s for request=%s",
        root_uri,
        request.model_dump_json(),
    )

    resolved: dict[str, ManifestRow] = {}
    for params in silver.dsgrid_requests(request.state, request.dsgrid):
        bronze = dsgrid_bronze.PROFILES[params.source]
        for kind, dataset_name in (
            ("metadata", bronze.metadata_dataset_name),
            ("timeseries", bronze.timeseries_dataset_name),
        ):
            try:
                row = resolve_manifest(
                    dataset_name=dataset_name,
                    params=params,
                    root_uri=root_uri,
                    as_of=as_of,
                )
            except KeyError as exc:
                msg = (
                    f"no {dataset_name} for state {request.state} -- ingest the "
                    "bronze slice before building silver"
                )
                raise RuntimeError(msg) from exc
            logger.info("resolved %s input: write_id=%s", dataset_name, row.write_id)
            resolved[f"{params.source}_{kind}"] = row
    return resolved


# The two silver tables, each its own unit of work. Held as *names* looked up on the
# package at call time rather than references captured at import, which would make
# the seam unpatchable in a test.
_SILVER_BUILDERS: dict[str, tuple[str, str]] = {
    "metadata": ("build_metadata_table", "write_metadata_silver"),
    "timeseries": ("build_timeseries_table", "write_timeseries_silver"),
}
SILVER_TABLES: tuple[str, ...] = tuple(_SILVER_BUILDERS)


def _silver_builder(
    kind: str,
) -> tuple[Callable[..., pl.DataFrame], Callable[..., ManifestRow]]:
    """
    The build + write pair for one silver table.

    ``kind`` crosses a task boundary as a string, so an unknown one fails here, before
    any bronze is read.

    Raises:
        PipelineValueError: If *kind* is not one of :data:`SILVER_TABLES`.
    """
    if kind not in _SILVER_BUILDERS:
        msg = f"unknown silver table {kind!r}; expected one of {SILVER_TABLES}"
        raise common.exceptions.PipelineValueError(msg)
    build, write = _SILVER_BUILDERS[kind]
    return getattr(silver, build), getattr(silver, write)


@prefect.task(timeout_seconds=SILVER_TIMEOUT_SECONDS)
def build_industrial_silver_table(
    kind: str,
    request: schema.IndustrialLoadRequestArgs,
    root_uri: str,
    writer: str,
    as_of: datetime.datetime | None = None,
) -> ManifestRow:
    """
    Stack one kind of dsgrid bronze into its silver table.

    A table at a time rather than both together: they read different bronze pairs and
    share nothing but the request, and the timeseries is much the more expensive
    (237k rows for DC against 27), so it gets its own task run to say so.

    Raises:
        PipelineValueError: If *kind* is not one of :data:`SILVER_TABLES`.
    """
    logger = prefect.get_run_logger()
    logger.info(
        "stacking dsgrid %s bronze under %s for request=%s",
        kind,
        root_uri,
        request.model_dump_json(),
    )
    build, write = _silver_builder(kind)
    row = write(
        build(request, root_uri, as_of),
        request,
        silver_root=root_uri,
        writer=writer,
    )
    logger.info("wrote %s -> %s", row.dataset_name, row.data_uri)
    return row


def build_industrial_silver(
    request: schema.IndustrialLoadRequestArgs,
    root_uri: str,
    writer: str,
    as_of: datetime.datetime | None = None,
) -> list[ManifestRow]:
    """Both silver tables for one state, each as its own task run."""
    futures = build_industrial_silver_table.map(
        list(SILVER_TABLES),
        prefect.unmapped(request),
        prefect.unmapped(root_uri),
        prefect.unmapped(writer),
        prefect.unmapped(as_of),
    )
    return [future.result() for future in futures]


# --------------------------------------------------------------------------- #
# 1. Residential bronze ingestion
# --------------------------------------------------------------------------- #


# prefect's flow overloads omit task_runner, so ty cannot match this call
@prefect.flow(  # ty:ignore[no-matching-overload]
    timeout_seconds=BUILDING_STOCK_FLOW_TIMEOUT_SECONDS,
    task_runner=ThreadPoolTaskRunner(max_workers=TASK_WORKERS),
)
def ingest_resstock(
    *,
    geographies: schema.LoadGeographies,
    resstock: schema.ResstockRequestArgs = schema.ResstockRequestArgs(),
    force_refresh: bool = False,
    root_uri: str = DEFAULT_ROOT,
    writer: str = DEFAULT_WRITER,
    as_of: datetime.datetime | None = None,
) -> None:
    """Fetch the ResStock metadata + PUMA timeseries for **every requested PUMA**
    from the public OEDI lake and write the bronze datasets under ``root_uri``.

    One HTTP request per building and every building in each PUMA is fetched, so a
    full PUMA takes a while; the count is logged before its fetches start. PUMAs are
    fetched several at a time, and each state's metadata file is read once ahead of
    them however many of its PUMAs are asked for. ``as_of`` is accepted for
    flow-signature consistency but unused: this flow resolves no inputs."""
    run_ts, scheduled_time, flow_id = _flow_prelude()
    logger = prefect.get_run_logger()
    root_uri = _resolve_root(root_uri)

    write_flow_manifest_start(
        flow_id=flow_id,
        flow_name="ingest_resstock",
        writer=writer,
        root_uri=root_uri,
        scheduled_time=scheduled_time,
        start_time=run_ts,
    )
    # Owned out here so the except branch can still record what was discovered
    # before the failure.
    failed: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    # Set once a terminal status is written, so the finally can tell a run that
    # ended from one that was stopped.
    recorded = False
    try:
        rows = _ingest_resstock_metadata(
            geographies, resstock, root_uri, writer, force_refresh, rejected
        )
        # The metadata phase is where a bad code shows. Fetch the timeseries only
        # for the PUMAs it could actually write, so one typo costs its own PUMA and
        # nothing else.
        surviving = _surviving(geographies, rejected, "resstock")
        _report_rejected(rejected, len(geographies.pumas), surviving)
        _refuse_if_nothing_left(surviving, rejected)
        rows.extend(
            _ingest_each_unit(
                fetch_resstock_bronze,
                surviving,
                resstock,
                root_uri,
                writer,
                force_refresh,
                failed,
            )
        )
        update_flow_manifest(
            flow_id=flow_id,
            root_uri=root_uri,
            status=FlowStatus.COMPLETED,
            metadata={
                "geographies": _run_geographies(geographies),
                "output_write_ids": [row.write_id for row in rows],
                # Recorded, not just logged: a later run reads these back and
                # stops re-asking for files the release has already refused.
                "failed_buildings": failed,
                # The bad codes, so the run that skipped them says so durably and
                # not only in a log line. Never read back to skip a PUMA -- see
                # failures.describe_rejected.
                failures.REJECTED_PUMAS_KEY: rejected,
            },
        )
        recorded = True
    except Exception as exc:
        update_flow_manifest(
            flow_id=flow_id,
            root_uri=root_uri,
            status=FlowStatus.FAILED,
            error=exc,
            # Recorded on the way out too: these are the buildings the release
            # refused before the failure, and dropping them because the run died
            # would leave the next run asking for the same dead files again.
            metadata={
                "failed_buildings": failed,
                failures.REJECTED_PUMAS_KEY: rejected,
            },
        )
        recorded = True
        raise
    finally:
        # Cancellation is a BaseException, so it reaches here and not the except
        # above. Without this the manifest is left saying "running" and the run's
        # records die with the process.
        if not recorded:
            _record_interruption(
                flow_id,
                root_uri,
                {
                    "failed_buildings": failed,
                    failures.REJECTED_PUMAS_KEY: rejected,
                },
            )
    logger.info("finished ResStock bronze ingest")


# --------------------------------------------------------------------------- #
# 2. Commercial bronze ingestion
# --------------------------------------------------------------------------- #


# prefect's flow overloads omit task_runner, so ty cannot match this call
@prefect.flow(  # ty:ignore[no-matching-overload]
    timeout_seconds=BUILDING_STOCK_FLOW_TIMEOUT_SECONDS,
    task_runner=ThreadPoolTaskRunner(max_workers=TASK_WORKERS),
)
def ingest_comstock(
    *,
    geographies: schema.LoadGeographies,
    comstock: schema.ComstockRequestArgs = schema.ComstockRequestArgs(),
    force_refresh: bool = False,
    root_uri: str = DEFAULT_ROOT,
    writer: str = DEFAULT_WRITER,
    as_of: datetime.datetime | None = None,
) -> None:
    """Fetch the ComStock per-PUMA metadata + timeseries for **every requested PUMA**
    from the public OEDI lake and write the bronze datasets under ``root_uri``.

    One HTTP request per building and every building in each PUMA is fetched; the
    count is logged before its fetches start. ``as_of`` is accepted for flow-signature
    consistency but unused: this flow resolves no inputs."""
    run_ts, scheduled_time, flow_id = _flow_prelude()
    logger = prefect.get_run_logger()
    root_uri = _resolve_root(root_uri)

    write_flow_manifest_start(
        flow_id=flow_id,
        flow_name="ingest_comstock",
        writer=writer,
        root_uri=root_uri,
        scheduled_time=scheduled_time,
        start_time=run_ts,
    )
    # Owned out here so the except branch can still record what was discovered
    # before the failure.
    failed: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    # Set once a terminal status is written, so the finally can tell a run that
    # ended from one that was stopped.
    recorded = False
    try:
        rows = _ingest_comstock_metadata(
            geographies, comstock, root_uri, writer, force_refresh, rejected
        )
        surviving = _surviving(geographies, rejected, "comstock")
        _report_rejected(rejected, len(geographies.pumas), surviving)
        _refuse_if_nothing_left(surviving, rejected)
        rows.extend(
            _ingest_each_unit(
                fetch_comstock_bronze,
                surviving,
                comstock,
                root_uri,
                writer,
                force_refresh,
                failed,
            )
        )
        update_flow_manifest(
            flow_id=flow_id,
            root_uri=root_uri,
            status=FlowStatus.COMPLETED,
            metadata={
                "geographies": _run_geographies(geographies),
                "output_write_ids": [row.write_id for row in rows],
                # Recorded, not just logged: a later run reads these back and
                # stops re-asking for files the release has already refused.
                "failed_buildings": failed,
                # The bad codes, so the run that skipped them says so durably and
                # not only in a log line. Never read back to skip a PUMA -- see
                # failures.describe_rejected.
                failures.REJECTED_PUMAS_KEY: rejected,
            },
        )
        recorded = True
    except Exception as exc:
        update_flow_manifest(
            flow_id=flow_id,
            root_uri=root_uri,
            status=FlowStatus.FAILED,
            error=exc,
            # Recorded on the way out too: these are the buildings the release
            # refused before the failure, and dropping them because the run died
            # would leave the next run asking for the same dead files again.
            metadata={
                "failed_buildings": failed,
                failures.REJECTED_PUMAS_KEY: rejected,
            },
        )
        recorded = True
        raise
    finally:
        # Cancellation is a BaseException, so it reaches here and not the except
        # above. Without this the manifest is left saying "running" and the run's
        # records die with the process.
        if not recorded:
            _record_interruption(
                flow_id,
                root_uri,
                {
                    "failed_buildings": failed,
                    failures.REJECTED_PUMAS_KEY: rejected,
                },
            )
    logger.info("finished ComStock bronze ingest")


# --------------------------------------------------------------------------- #
# 3. Industrial bronze ingestion
# --------------------------------------------------------------------------- #


# prefect's flow overloads omit task_runner, so ty cannot match this call
@prefect.flow(  # ty:ignore[no-matching-overload]
    timeout_seconds=DSGRID_FLOW_TIMEOUT_SECONDS,
    task_runner=ThreadPoolTaskRunner(max_workers=TASK_WORKERS),
)
def ingest_dsgrid(
    *,
    geographies: schema.LoadGeographies,
    dsgrid: schema.DsgridRequestArgs = schema.DsgridRequestArgs(),
    force_refresh: bool = False,
    root_uri: str = DEFAULT_ROOT,
    writer: str = DEFAULT_WRITER,
    as_of: datetime.datetime | None = None,
) -> None:
    """Reconstruct the requested dsgrid industrial source file(s) for the geographies'
    *states* and write their bronze datasets under ``root_uri``.

    Only the states matter -- dsgrid is published per state and covers every county,
    so one ingest serves every PUMA in it and a run's PUMAs collapse to their distinct
    states here. ``as_of`` is accepted for flow-signature consistency but unused: this
    flow resolves no inputs."""
    run_ts, scheduled_time, flow_id = _flow_prelude()
    logger = prefect.get_run_logger()
    root_uri = _resolve_root(root_uri)

    write_flow_manifest_start(
        flow_id=flow_id,
        flow_name="ingest_dsgrid",
        writer=writer,
        root_uri=root_uri,
        scheduled_time=scheduled_time,
        start_time=run_ts,
    )
    # Owned out here so the except branch can still record what was discovered
    # before the failure.
    failed: list[dict[str, str]] = []
    recorded = False
    try:
        # No metadata phase to reject a code here: dsgrid is published per state
        # and a PUMA GISJOIN only has to be well-formed for its state to be read
        # off the FIPS prefix. So a code that no building-stock release has still
        # names a real state, and this flow ingests it.
        rows = _ingest_each_unit(
            fetch_dsgrid_bronze,
            geographies.pumas,
            dsgrid,
            root_uri,
            writer,
            force_refresh,
            failed,
            by_state=True,
        )
        update_flow_manifest(
            flow_id=flow_id,
            root_uri=root_uri,
            status=FlowStatus.COMPLETED,
            metadata={
                "geographies": _run_geographies(geographies),
                "output_write_ids": [row.write_id for row in rows],
                "failed_buildings": failed,
            },
        )
        recorded = True
    except Exception as exc:
        update_flow_manifest(
            flow_id=flow_id,
            root_uri=root_uri,
            status=FlowStatus.FAILED,
            error=exc,
            # Recorded on the way out too: these are the buildings the release
            # refused before the failure, and dropping them because the run died
            # would leave the next run asking for the same dead files again.
            metadata={"failed_buildings": failed},
        )
        recorded = True
        raise
    finally:
        # Cancellation is a BaseException, so it reaches here and not the except
        # above. Without this the manifest is left saying "running" and the run's
        # records die with the process.
        if not recorded:
            _record_interruption(flow_id, root_uri, {"failed_buildings": failed})
    logger.info("finished dsgrid bronze ingest")


# --------------------------------------------------------------------------- #
# 4. Industrial silver (both dsgrid sources stacked by kind)
# --------------------------------------------------------------------------- #


@prefect.flow(timeout_seconds=SILVER_FLOW_TIMEOUT_SECONDS)
def industrial_load_silver(
    *,
    geographies: schema.LoadGeographies,
    dsgrid: schema.DsgridRequestArgs = schema.DsgridRequestArgs(),
    root_uri: str = DEFAULT_ROOT,
    writer: str = DEFAULT_WRITER,
    as_of: datetime.datetime | None = None,
) -> None:
    """Stack the requested dsgrid sources by kind: their metadata bronze into one
    silver table and their timeseries bronze into another, so four bronze tables
    become two silver ones **per state**. Bronze must already be ingested.

    Takes the same ``geographies`` as the bronze ingests for a consistent run form,
    but uses only their **states**: dsgrid publishes per state and covers every
    county, so the tables are keyed by state and the PUMA does not narrow them. Two
    PUMAs in one state produce one table per kind, not two.

    Every state's bronze is resolved **before** any table is built, so a run missing
    one state's bronze fails without having written a table for the others.

    ``as_of`` (default: the scheduled time) bounds both the lineage resolve and the
    data reads to the same point in time, so each pair is consistent."""
    run_ts, scheduled_time, flow_id = _flow_prelude()
    logger = prefect.get_run_logger()
    root_uri = _resolve_root(root_uri)
    resolved_as_of = as_of or scheduled_time
    # dsgrid is state-published; the PUMAs are only how the states are supplied.
    requests = [
        schema.IndustrialLoadRequestArgs(state=state, dsgrid=dsgrid)
        for state in geographies.states
    ]

    write_flow_manifest_start(
        flow_id=flow_id,
        flow_name="industrial_load_silver",
        writer=writer,
        root_uri=root_uri,
        scheduled_time=scheduled_time,
        start_time=run_ts,
    )
    recorded = False
    try:
        input_ids: list[str] = []
        resolved: dict[str, dict[str, str]] = {}
        for request in requests:
            inputs = resolve_industrial_inputs(request, root_uri, resolved_as_of)
            input_ids.extend(row.write_id for row in inputs.values())
            # Keyed by state: with several states in a run, ``industrial_metadata``
            # alone would name four different bronze tables.
            resolved[request.state] = {
                name: row.params_json for name, row in inputs.items()
            }
        update_flow_manifest(
            flow_id=flow_id,
            root_uri=root_uri,
            input_ids=input_ids,
            metadata={"resolved_inputs": resolved},
        )

        rows: list[ManifestRow] = []
        for index, request in enumerate(requests, start=1):
            logger.info("state %d/%d: %s", index, len(requests), request.state)
            rows.extend(
                build_industrial_silver(request, root_uri, writer, resolved_as_of)
            )
        update_flow_manifest(
            flow_id=flow_id,
            root_uri=root_uri,
            status=FlowStatus.COMPLETED,
            metadata={
                # dsgrid is state-published, so the PUMAs are only how the states
                # were supplied -- recorded anyway, since it is what was typed.
                "geographies": _run_geographies(geographies),
                "output_write_ids": [row.write_id for row in rows],
            },
        )
        recorded = True
    except Exception as exc:
        # No failed_buildings here: silver reads bronze off disk, so it has no
        # per-building unit to tolerate a failure of.
        update_flow_manifest(
            flow_id=flow_id, root_uri=root_uri, status=FlowStatus.FAILED, error=exc
        )
        recorded = True
        raise
    finally:
        # Cancellation is a BaseException, so it reaches here and not the except
        # above; without it the manifest is left saying "running" forever.
        if not recorded:
            _record_interruption(flow_id, root_uri, {})
    logger.info("finished industrial silver stack")


# --------------------------------------------------------------------------- #
# 5. Full pipeline (three bronze ingests, then the silver build)
# --------------------------------------------------------------------------- #


# prefect's flow overloads omit task_runner, so ty cannot match this call
@prefect.flow(  # ty:ignore[no-matching-overload]
    timeout_seconds=PIPELINE_TIMEOUT_SECONDS,
    # Wide enough for every parent in flight *plus* the whole per-building HTTP
    # budget. It used to be 3 -- one slot per source -- which left the mapped
    # fetches nothing to run on. See TASK_WORKERS.
    task_runner=ThreadPoolTaskRunner(max_workers=TASK_WORKERS),
)
def run_load_pipeline(
    *,
    geographies: schema.LoadGeographies,
    resstock: schema.ResstockRequestArgs = schema.ResstockRequestArgs(),
    comstock: schema.ComstockRequestArgs = schema.ComstockRequestArgs(),
    dsgrid: schema.DsgridRequestArgs = schema.DsgridRequestArgs(),
    force_refresh: bool = False,
    root_uri: str = DEFAULT_ROOT,
    writer: str = DEFAULT_WRITER,
    as_of: datetime.datetime | None = None,
) -> None:
    """
    End-to-end: for each requested PUMA ingest the residential, commercial and
    industrial bronze **concurrently**, then stack the industrial bronze by kind into
    two silver tables per state. All datasets land under ``root_uri``.

    The three sources are independent, so they are submitted as **tasks on this
    flow's task runner** rather than called as subflows: Prefect runs subflows one
    after another, so a subflow can never overlap however its work is written. The
    standalone ``ingest_*`` flows wrap the same tasks, for running one source alone.

    The overlap is within a PUMA and across them -- every source of every PUMA in the
    window is submitted before anything is waited on -- so the runner is sized for
    those parents *plus* the per-building fan-out they map (see ``TASK_WORKERS``).

    dsgrid is submitted only for a PUMA whose **state** this run has not reached yet:
    it is published per state, so a second PUMA of that state would submit a task
    whose whole job is to find the ledger already satisfied.

    A pipeline run emits **one** flow manifest rather than four, so this flow records
    every bronze ``write_id`` itself.
    """
    run_ts, scheduled_time, flow_id = _flow_prelude()
    logger = prefect.get_run_logger()
    root_uri = _resolve_root(root_uri)

    # Owned outside the try so the except branch can still record what this run
    # discovered the release cannot serve.
    failed: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    # Set once a terminal status is written, so the finally can tell a run that
    # ended from one that was stopped.
    recorded = False

    write_flow_manifest_start(
        flow_id=flow_id,
        flow_name="run_load_pipeline",
        writer=writer,
        root_uri=root_uri,
        scheduled_time=scheduled_time,
        start_time=run_ts,
    )
    try:
        logger.info(
            "input params -> geographies=%s resstock=%s comstock=%s dsgrid=%s",
            geographies.model_dump_json(),
            resstock.model_dump_json(),
            comstock.model_dump_json(),
            dsgrid.model_dump_json(),
        )

        # Both metadata phases first: ResStock's one read per state written per
        # PUMA, ComStock's one per PUMA. The fetches below read their building ids
        # out of these, so neither can overlap them -- and hoisting the residential
        # one keeps a state's 54 MB file to one download however many of its PUMAs
        # the run covers.
        rows: list[ManifestRow] = _ingest_resstock_metadata(
            geographies, resstock, root_uri, writer, force_refresh, rejected
        )
        rows.extend(
            _ingest_comstock_metadata(
                geographies, comstock, root_uri, writer, force_refresh, rejected
            )
        )

        # Both metadata phases have now had their say, so this is the run's full
        # picture of which codes the releases have no file for. A bad one costs its
        # own PUMA from here on and nothing else -- which is the whole point of
        # rejecting a PUMA rather than the run.
        resstock_pumas = _surviving(geographies, rejected, "resstock")
        comstock_pumas = _surviving(geographies, rejected, "comstock")
        serviceable = _still_serviceable(geographies, rejected)
        _report_rejected(rejected, len(geographies.pumas), serviceable)
        _refuse_if_nothing_left(serviceable, rejected)

        ingested_states: set[str] = set()
        pumas = serviceable
        for start in range(0, len(pumas), PUMAS_IN_FLIGHT):
            window = pumas[start : start + PUMAS_IN_FLIGHT]
            logger.info(
                "step 1/2, PUMA %d-%d of %d: bronze, concurrently (%s)",
                start + 1,
                start + len(window),
                len(pumas),
                ", ".join(g.puma_gisjoin for g in window),
            )
            # Submit every source of every PUMA in the window before waiting on any
            # -- the whole point of tasks rather than subflows. The pool is sized so
            # their mapped children still run while the parents sit here blocked.
            submitted: list[tuple[str, str, object]] = []
            for geography in window:
                # Per source: a code ResStock has no file for may still be one
                # ComStock publishes, and vice versa. Submitting only the halves
                # that can succeed is what keeps one release's gap from dropping
                # the other release's data.
                if geography in resstock_pumas:
                    submitted.append(
                        (
                            "ResStock",
                            geography.puma_gisjoin,
                            fetch_resstock_bronze.submit(
                                geography, resstock, root_uri, writer, force_refresh
                            ),
                        )
                    )
                if geography in comstock_pumas:
                    submitted.append(
                        (
                            "ComStock",
                            geography.puma_gisjoin,
                            fetch_comstock_bronze.submit(
                                geography, comstock, root_uri, writer, force_refresh
                            ),
                        )
                    )
                # dsgrid is per state, so only the first PUMA of each state submits
                # it: a second would hold a worker to re-reach the same answer.
                if geography.state not in ingested_states:
                    ingested_states.add(geography.state)
                    submitted.append(
                        (
                            "dsgrid",
                            geography.state,
                            fetch_dsgrid_bronze.submit(
                                geography, dsgrid, root_uri, writer, force_refresh
                            ),
                        )
                    )
            for source, where, future in submitted:
                # ``.result()`` re-raises, so a source that fails outright still
                # fails the run. Individual buildings are different: those are
                # tolerated inside the fetch and come back here as records.
                source_rows, source_failed = future.result()  # ty:ignore[unresolved-attribute]
                logger.info(
                    "%s bronze for %s: %d dataset(s), %d building(s) unavailable",
                    source,
                    where,
                    len(source_rows),
                    len(source_failed),
                )
                rows.extend(source_rows)
                failed.extend(source_failed)
        update_flow_manifest(
            flow_id=flow_id,
            root_uri=root_uri,
            metadata={
                "geographies": _run_geographies(geographies),
                "output_write_ids": [row.write_id for row in rows],
                "failed_buildings": failed,
                # The codes this run skipped. ``geographies`` above stays what was
                # *asked* for, so the skip reads as the difference between the two
                # rather than the request quietly shrinking.
                failures.REJECTED_PUMAS_KEY: rejected,
            },
        )

        logger.info("step 2/2: industrial silver stack")
        # Deliberately NOT threading ``as_of``: the bronze above was just written
        # with ``write_time = now`` and silver resolves ``write_time <= as_of``, so a
        # past ``as_of`` would exclude this run's own output and fail after paying for
        # the hour-long fetches. The subflow defaults to its scheduled time.
        # The PUMAs that survived, not the whole request: silver is keyed by state,
        # and a state reached only by a bad code has no dsgrid bronze above to stack.
        industrial_load_silver(
            geographies=schema.LoadGeographies(pumas=serviceable),
            dsgrid=dsgrid,
            root_uri=root_uri,
            writer=writer,
        )

        update_flow_manifest(
            flow_id=flow_id, root_uri=root_uri, status=FlowStatus.COMPLETED
        )
        recorded = True
    except Exception as exc:
        update_flow_manifest(
            flow_id=flow_id,
            root_uri=root_uri,
            status=FlowStatus.FAILED,
            error=exc,
            # Recorded on the way out too: these are the buildings the release
            # refused before the failure, and dropping them because the run died
            # would leave the next run asking for the same dead files again.
            metadata={
                "failed_buildings": failed,
                failures.REJECTED_PUMAS_KEY: rejected,
            },
        )
        recorded = True
        raise
    finally:
        # Cancellation is a BaseException, so it reaches here and not the except
        # above. Without this the manifest is left saying "running" and the run's
        # records die with the process.
        if not recorded:
            _record_interruption(
                flow_id,
                root_uri,
                {
                    "failed_buildings": failed,
                    failures.REJECTED_PUMAS_KEY: rejected,
                },
            )
    logger.info(
        "finished load pipeline: %d of %d requested PUMA(s) of bronze x3 -> "
        "industrial silver x2 for %d state(s)%s",
        len(serviceable),
        len(geographies.pumas),
        len({g.state for g in serviceable}),
        (
            f"; {len(rejected)} PUMA(s) skipped as absent from the release "
            f"({failures.describe_rejected(rejected)})"
            if rejected
            else ""
        ),
    )
