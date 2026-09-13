"""
Shared machinery for the OEDI building-stock bronze modules

ResStock (residential) and ComStock (commercial) ingest the same *End-Use Load
Profiles for the U.S. Building Stock* release family from the public OEDI data
lake, in the same shape: a per-building metadata / annual-results parquet, and
per-building 15-min ``timeseries_individual_buildings`` files assembled a whole
PUMA at a time.

Held here: the OEDI location constants, the fetch-with-retry helper, the shared
request fields, the per-building timeseries URL, the (identical) timeseries bronze
schema, and the PUMA assembly + ingest. Each source module keeps its own metadata
schema, building-id resolution, catalog entry and writes. Concurrency is not here:
the orchestrator injects it (see ``fetch_buildings``).
"""

from __future__ import annotations
import dataclasses
import datetime as dt
import io
import json
import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol
import httpx
import patito as pt
import polars as pl
import pydantic
import common.exceptions
import common.models
from external_data.load_pipeline import failures
from common.frames import BaseDataFrameSchema
from common.storage import columnar
from common.storage.manifest import ManifestRow, scan_manifest
from external_data.load_pipeline import coverage
from external_data.load_pipeline.retry import retry_wait

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration (shared)
# --------------------------------------------------------------------------- #

# Public OEDI data lake (no account, API key, or login required)
BASE_URL = "https://oedi-data-lake.s3.amazonaws.com"
OEDI_PREFIX = "nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock"

# Latest publication year directory (both sources' current release)
RELEASE_YEAR = 2025

DEFAULT_BASE_DIR = "bronze"
DEFAULT_STATE = "DC"
DEFAULT_UPGRADE = 0
DEFAULT_PUMA = "G11000101"  # Washington, DC

STATE_PATTERN = r"^[A-Z]{2}$"
# "G" + 2-digit state FIPS + "0" + 5-digit PUMA code, 2010-census vintage to match
# the AMY2018 releases.
PUMA_PATTERN = r"^G\d{8}$"


# --------------------------------------------------------------------------- #
# Request base
# --------------------------------------------------------------------------- #


class OediBuildingStockRequestBase(common.models.FrozenModel, extra="forbid"):
    """
    Shared release + geography fields for OEDI building-stock requests

    Each source subclass adds ``source_dataset`` (its release directory, as the
    field default) plus any source-specific geography fields.
    """

    release_year: int = pydantic.Field(
        default=RELEASE_YEAR,
        description="OEDI publication year directory (e.g. 2025)",
    )
    upgrade: int = pydantic.Field(
        default=DEFAULT_UPGRADE,
        ge=0,
        description="upgrade / measure-package id; 0 is the un-retrofit baseline",
    )
    state: str = pydantic.Field(
        default=DEFAULT_STATE,
        pattern=STATE_PATTERN,
        description="two-letter state code the file is partitioned under",
    )


# What the orchestrator injects to fetch a PUMA's buildings concurrently: given the
# ids, return the assembled table and the ids it actually got. The table is ``None``
# exactly when nothing was fetched, spelled out in the type so a caller cannot read
# the frame without having checked.
BuildingFetcher = Callable[[Sequence[int]], tuple["pl.DataFrame | None", list[int]]]


class _PumaRequestLike(Protocol):
    """The request attributes the shared PUMA helpers rely on."""

    release_year: int
    source_dataset: str
    upgrade: int
    state: str
    puma_gisjoin: str

    def building_timeseries_url(self, bldg_id: int) -> str: ...


def building_timeseries_url(args: _PumaRequestLike, bldg_id: int) -> str:
    """
    Full HTTPS URL of a single building's ``timeseries_individual_buildings``
    parquet (identical layout for ResStock and ComStock)
    """
    return (
        f"{BASE_URL}/{OEDI_PREFIX}/{args.release_year}/{args.source_dataset}/"
        f"timeseries_individual_buildings/by_state/upgrade={args.upgrade}/"
        f"state={args.state}/{bldg_id}-{args.upgrade}.parquet"
    )


# --------------------------------------------------------------------------- #
# Fetch (raw)
# --------------------------------------------------------------------------- #


def download_object(
    url: str,
    *,
    max_retries: int = 3,
    backoff_seconds: float = 2.0,
    client: httpx.Client | None = None,
) -> bytes:
    """
    Download an object from the public OEDI data lake and return its bytes

    A generic "GET bytes with retry" helper: used for both parquet files and the
    S3 XML list-objects responses. No credentials are required; the bucket is
    fully public.
    """
    owns_client = client is None
    client = (
        client
        if client is not None
        else httpx.Client(timeout=120, follow_redirects=True)
    )
    last_exc: Exception | None = None
    try:
        for attempt in range(1, max_retries + 1):
            logger.info(
                "requesting OEDI object (%s, attempt=%d/%d)", url, attempt, max_retries
            )
            try:
                response = client.get(url)
            except httpx.RequestError as exc:
                last_exc = exc
                if attempt < max_retries:
                    wait = backoff_seconds * (2 ** (attempt - 1))
                    logger.warning("request failed (%s); retrying in %.1fs", exc, wait)
                    time.sleep(wait)
                continue

            # 429 is transient like 5xx. OEDI publishes no rate limit, so a
            # Retry-After header is the only budget figure the server volunteers:
            # honour it over our own backoff.
            if response.status_code == 429 or response.status_code >= 500:
                last_exc = httpx.HTTPStatusError(
                    f"transient status {response.status_code}",
                    request=response.request,
                    response=response,
                )
                if attempt < max_retries:
                    wait = retry_wait(response, backoff_seconds, attempt)
                    logger.warning(
                        "transient status %d; retrying in %.1fs",
                        response.status_code,
                        wait,
                    )
                    time.sleep(wait)
                continue
            # Other 4xx are permanent against an object store: 403/404 mean the
            # key is not readable, 400 that the request is malformed. Classified
            # here, at the raise site -- see load_pipeline.failures.
            permanent = failures.as_permanent_oedi(
                response.status_code, url, response.text
            )
            if permanent is not None:
                raise permanent

            return response.content
    finally:
        if owns_client:
            client.close()

    raise common.exceptions.PipelineError(
        f"failed to download {url} after {max_retries} attempts"
    ) from last_exc


# --------------------------------------------------------------------------- #
# Parse / shape helpers
# --------------------------------------------------------------------------- #


def read_curated(
    data: bytes, raw_to_column: dict[str, str], label: str
) -> pl.DataFrame:
    """
    Project a parquet blob to the curated raw columns and rename them to the
    clean bronze field names, failing loud on any missing column
    """
    available = set(pl.read_parquet_schema(io.BytesIO(data)))
    missing = [c for c in raw_to_column if c not in available]
    if missing:
        msg = (
            f"OEDI {label} download is missing expected column(s) {missing}; "
            f"got {len(available)} columns"
        )
        raise common.exceptions.PipelineValueError(msg)

    frame = pl.read_parquet(io.BytesIO(data), columns=list(raw_to_column))
    return frame.rename(raw_to_column)


def shape_to_schema(
    frame: pl.DataFrame,
    schema: type[BaseDataFrameSchema],
    columns: list[str],
    key_columns: list[str],
) -> pl.DataFrame:
    """
    Cast a curated frame to *schema*'s declared dtypes, order/sort it, and
    validate -- so the schema definition is the single source of dtype truth
    """
    shaped = schema.DataFrame(frame.select(columns)).cast().sort(key_columns)
    schema.validate(shaped)
    return shaped


def resolve_root_uri(root_uri: str | Path) -> str:
    if "://" not in str(root_uri):
        return str(Path(root_uri).resolve())
    return str(root_uri)


# --------------------------------------------------------------------------- #
# Timeseries bronze schema (shared -- identical for ResStock and ComStock)
# --------------------------------------------------------------------------- #


TIMESERIES_COLUMNS = [
    "timestamp",
    "bldg_id",
    "state",
    "puma_gisjoin",
    "electricity_total_kwh",
    "electricity_cooling_kwh",
    "electricity_heating_kwh",
]
TIMESERIES_KEY_COLUMNS = ["timestamp", "bldg_id"]
# No partition columns: a write is one PUMA, so a ``state=/puma=`` directory would
# hold a single file and repeat what the key says. The manifest is the index -- the
# same choice the climate pipeline makes per point.


class OediBuildingStockTimeseriesSchema(BaseDataFrameSchema):
    """
    One row per building x 15-min timestep of the individual-building timeseries
    (assembled a whole PUMA at a time)
    """

    timestamp: dt.datetime = pt.Field(
        dtype=pl.Datetime(time_unit="us"),
        description="interval-ending timestamp (local standard time, AMY2018; naive)",
    )
    bldg_id: int = pt.Field(
        dtype=pl.Int64,
        description="building model id; join key across buildings in the PUMA",
    )
    state: str = pt.Field(dtype=pl.String, description="two-letter state code")
    puma_gisjoin: str = pt.Field(
        dtype=pl.String, description="NHGIS PUMA GISJOIN the buildings belong to"
    )
    electricity_total_kwh: float = pt.Field(
        dtype=pl.Float32,
        description="total whole-building electricity for the interval (kWh)",
    )
    electricity_cooling_kwh: float = pt.Field(
        dtype=pl.Float32, description="cooling electricity for the interval (kWh)"
    )
    electricity_heating_kwh: float = pt.Field(
        dtype=pl.Float32,
        description="electric-heating electricity for the interval (kWh)",
    )


# --------------------------------------------------------------------------- #
# Reading a PUMA back
# --------------------------------------------------------------------------- #


def check_complete(
    table: pl.DataFrame,
    expected_buildings: int,
    label: str,
    *,
    require_complete: bool = False,
) -> None:
    """
    Compare what a PUMA holds against what it should, and say so on a shortfall.

    Load data is summed and peaked, so a PUMA holding 943 of its 946 buildings
    understates demand by ~0.3% -- and nothing in the data, the schema or the row
    count reveals it. A short read is not an error a caller notices; it is a
    plausible number.

    Warns rather than raises, because a shortfall is sometimes the correct state: a
    PUMA written short over a building the release cannot serve is as complete as it
    will ever get. ``require_complete`` is for a caller that would rather fail than
    publish a number it cannot stand behind.
    """
    held = table["bldg_id"].n_unique()
    if held >= expected_buildings:
        return
    missing = expected_buildings - held
    msg = (
        f"{label} holds {held} of {expected_buildings} building(s) -- "
        f"{missing} missing ({missing / expected_buildings:.1%}). Any sum or peak "
        "read off this understates demand by roughly that share. Re-run the ingest "
        "to fill the gap, or check the flow manifests' failed_buildings for "
        "buildings the release cannot serve."
    )
    if require_complete:
        raise common.exceptions.PipelineValueError(msg)
    logger.warning(msg)


# Legacy chunk keys carried these; a whole-PUMA key does not, so their presence is
# how a read tells an old write from a current one.
_CHUNK_KEY_FIELDS = ("first_bldg_id", "last_bldg_id")


def read_puma_timeseries(
    schema: type[BaseDataFrameSchema],
    dataset_name: str,
    base_params: pydantic.BaseModel,
    root_uri: str | Path,
    *,
    as_of: dt.datetime | None = None,
    expected_buildings: int | None = None,
    require_complete: bool = False,
) -> pl.DataFrame:
    """
    Read one PUMA's timeseries: one manifest scan, then the write it resolves to.

    A scan matched on the *base* fields (release, upgrade, state, PUMA) rather than a
    plain resolve, for the sake of writes already on disk: a current key is exactly
    ``base_params``, but earlier ones carry extra fields no caller can reconstruct (a
    building count, a chunk's id range). Matching on a subset reads every generation.

    A whole-PUMA write supersedes the chunks it was assembled from, so the two are
    never concatenated -- that would double every row they share.

    Raises:
        PipelineValueError: If nothing has been written for *base_params*.
    """
    wanted = base_params.model_dump(mode="json")
    whole: list[ManifestRow] = []
    chunks: list[ManifestRow] = []
    for row in scan_manifest(
        dataset_name=dataset_name, root_uri=str(root_uri), as_of=as_of
    ):
        params = json.loads(row.params_json)
        if any(params.get(field) != value for field, value in wanted.items()):
            continue
        if any(field in params for field in _CHUNK_KEY_FIELDS):
            chunks.append(row)
        else:
            whole.append(row)
    if whole:
        # Newest wins: every write is immutable, so re-ingesting a PUMA adds a
        # version rather than replacing one.
        rows = [max(whole, key=lambda row: row.write_time)]
        if chunks:
            logger.info(
                "%s: reading the whole-PUMA write, ignoring %d legacy chunk(s)",
                dataset_name,
                len(chunks),
            )
    else:
        rows = chunks
    if not rows:
        msg = (
            f"no {dataset_name} found under {root_uri} for "
            f"{base_params.model_dump_json()}; ingest the PUMA first"
        )
        raise common.exceptions.PipelineValueError(msg)
    table = pl.concat(
        [columnar.read_parquet(row.data_uri, schema) for row in rows],
        how="vertical",
    ).sort(TIMESERIES_KEY_COLUMNS)
    if expected_buildings is not None:
        check_complete(
            table,
            expected_buildings,
            f"{dataset_name} for {base_params.model_dump_json()}",
            require_complete=require_complete,
        )
    return table


# --------------------------------------------------------------------------- #
# PUMA timeseries assembly (concurrent per-building fetch -> typed table)
# --------------------------------------------------------------------------- #


def fetch_building_frame(
    bldg_id: int,
    args: _PumaRequestLike,
    client: httpx.Client,
    timeseries_raw_map: dict[str, str],
) -> pl.DataFrame:
    """
    Fetch and curate one building's timeseries, injecting state + PUMA (dtype
    casting is deferred to the combined shaping pass)

    **Public because it is the unit of concurrency.** The orchestrator maps it over a
    PUMA's building ids under a server-side limit, which is what buys per-building
    retries, per-building visibility and a budget retunable without a deploy.
    """
    data = download_object(args.building_timeseries_url(bldg_id), client=client)
    frame = read_curated(data, timeseries_raw_map, f"timeseries (bldg {bldg_id})")
    return frame.with_columns(
        pl.lit(args.state, dtype=pl.String).alias("state"),
        pl.lit(args.puma_gisjoin, dtype=pl.String).alias("puma_gisjoin"),
    )


def assemble_timeseries_table(frames: Sequence[pl.DataFrame]) -> pl.DataFrame:
    """
    Concat per-building frames into the typed bronze table.

    One shaping and validation pass over the whole PUMA rather than per building:
    cheaper, and the only way the sort means anything.
    """
    if not frames:
        msg = "no building frames to assemble"
        raise common.exceptions.PipelineValueError(msg)
    return shape_to_schema(
        pl.concat(frames, how="vertical"),
        OediBuildingStockTimeseriesSchema,
        TIMESERIES_COLUMNS,
        TIMESERIES_KEY_COLUMNS,
    )


@dataclasses.dataclass(frozen=True)
class PumaTimeseriesSource:
    """
    What differs between the two building-stock sources when a PUMA is ingested.

    The ingest itself is one behaviour, shared; only these differ -- the dataset it
    writes, and the functions that resolve ids, fetch serially and write. Same shape
    as :class:`~...dsgrid.bronze.DsgridProfile`: primitives in one place, behaviour
    derived from them.

    Callables rather than a module reference, so the seam is explicit and a test can
    substitute one without patching module globals.
    """

    dataset_name: str
    resolve_ids: Callable[..., list[int]]
    fetch_table: Callable[..., pl.DataFrame]
    write_table: Callable[..., ManifestRow]


def ingest_puma_timeseries(
    source: PumaTimeseriesSource,
    args: Any,
    root_uri: str | Path,
    writer: str,
    client: httpx.Client | None = None,
    write_time: dt.datetime | None = None,
    bldg_ids: list[int] | None = None,
    force_refresh: bool = False,
    fetch_buildings: BuildingFetcher | None = None,
    skip_bldg_ids: set[int] | None = None,
) -> list[ManifestRow]:
    """
    Full PUMA timeseries bronze step: resolve the building ids, fetch them all, and
    write the PUMA as **one** dataset.

    Returns a single-element list, so callers that once pooled several chunk rows read
    the result unchanged.

    **Every** building in the PUMA is fetched; there is deliberately no cap, since a
    partial slice would write under the same key as a complete one and nothing
    downstream could tell an understated aggregate from a real one.

    The single file costs two things, both accepted: the whole PUMA is assembled in
    memory before it can be written (1.2 GB of table for a full ComStock PUMA, with a
    process high-water mark several times that), and an interruption loses every
    fetch the PUMA had made. The first is what bounds how many PUMAs a run fetches at
    once -- see ``PUMAS_IN_FLIGHT`` in the flows.
    """
    root_uri = resolve_root_uri(root_uri)

    owns_client = client is None
    client = (
        client
        if client is not None
        else httpx.Client(timeout=120, follow_redirects=True)
    )
    try:
        if bldg_ids is None:
            bldg_ids = source.resolve_ids(args, client=client)
        # Buildings an earlier run recorded as permanently unavailable are dropped
        # before anything is fetched: asking again costs a request per run, forever,
        # for a file the release has already refused.
        skip = skip_bldg_ids or set()
        wanted = sorted({int(b) for b in bldg_ids} - skip)
        if not wanted:
            msg = (
                f"no buildings left to fetch for PUMA {args.puma_gisjoin}: all "
                f"{len(bldg_ids)} are recorded unavailable"
            )
            raise common.exceptions.PipelineValueError(msg)
        # Logged before the fan-out is spent, so an unexpected count is the
        # operator's cue to stop the run.
        logger.info(
            "PUMA %s (state %s): fetching %d building(s) into one write",
            args.puma_gisjoin,
            args.state,
            len(wanted),
        )

        # The write key **is** the request -- the PUMA, its release, its upgrade --
        # and says nothing about what the fetch returned, so it is a key a reader can
        # build without knowing the building count in advance. A shortfall is
        # reported instead: logged here, reported by the flow against the metadata's
        # building list, and countable from the data itself.
        covered = (
            None
            if force_refresh
            else coverage.covered_write(source.dataset_name, args, root_uri)
        )
        if covered is not None:
            logger.info("reusing %s -> %s", source.dataset_name, covered.data_uri)
            return [covered]

        # ``fetch_buildings`` is how the orchestrator supplies concurrency without
        # this package importing Prefect: it maps fetch_building_frame over the ids
        # under a server-side limit and reports which it got. Absent, the serial
        # reference path runs and gets all or nothing.
        if fetch_buildings is not None:
            table, fetched = fetch_buildings(wanted)
            # Both halves checked so the type narrows: testing ``fetched`` alone
            # leaves ``table`` a ``DataFrame | None`` all the way into the write.
            if table is None or not fetched:
                logger.warning(
                    "PUMA %s: nothing fetched; leaving it unwritten",
                    args.puma_gisjoin,
                )
                return []
            if len(fetched) != len(wanted):
                # The write is short, and its key cannot say so. The failures land
                # on the run's flow manifest instead, and a later run excludes them
                # before computing this same key -- so it reuses this write rather
                # than re-requesting files the release has already refused.
                logger.warning(
                    "PUMA %s: writing %d of %d building(s); the rest were refused",
                    args.puma_gisjoin,
                    len(fetched),
                    len(wanted),
                )
        else:
            table = source.fetch_table(args, client, wanted)
        row = source.write_table(
            table,
            params=args,
            root_uri=root_uri,
            writer=writer,
            write_time=write_time,
        )
        logger.info(
            "PUMA %s: %d building(s), %d row(s) -> %s",
            args.puma_gisjoin,
            table["bldg_id"].n_unique(),
            table.height,
            row.data_uri,
        )
        return [row]
    finally:
        if owns_client:
            client.close()


def fetch_puma_timeseries_table(
    args: _PumaRequestLike,
    client: httpx.Client,
    bldg_ids: list[int],
    *,
    timeseries_raw_map: dict[str, str],
) -> pl.DataFrame:
    """
    Fetch the given buildings' timeseries **in order** and assemble the typed table.

    Deliberately serial: a pool here would be concurrency inside the domain package,
    which the layering rule forbids and which costs per-building retries, progress
    and a retunable budget. The orchestrator maps :func:`fetch_building_frame` and
    injects it as ``fetch_buildings`` instead; this is the reference path a
    non-Prefect caller (a test, a probe, ``__main__``) gets.

    Fail-loud on any building, wrapped so the error names the PUMA it aborted rather
    than surfacing as a bare per-building one.
    """
    if not bldg_ids:
        msg = "no building ids to fetch timeseries for (empty bldg_ids)"
        raise common.exceptions.PipelineValueError(msg)
    try:
        frames = [
            fetch_building_frame(b, args, client, timeseries_raw_map) for b in bldg_ids
        ]
    except Exception as exc:
        raise common.exceptions.PipelineError(
            f"failed to assemble the {len(bldg_ids)}-building timeseries for "
            f"PUMA {args.puma_gisjoin} (state {args.state}): {exc}"
        ) from exc
    return assemble_timeseries_table(frames)
