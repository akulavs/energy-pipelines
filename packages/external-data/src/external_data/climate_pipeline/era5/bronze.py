"""
ERA5-Land timeseries -- bronze ingestion.

Fetches ERA5-Land timeseries from the Copernicus CDS and writes **one single-file
bronze dataset per 0.1 deg grid node**, each with its own manifest entry. No
physical partitions -- the manifest is the spatial index (see
``climate_pipeline.point_manifest``), which lets a re-run skip nodes it already
has and lets concurrent fetches write without contending.

Two request types from ``climate_pipeline.schema`` play distinct roles:

- :class:`~external_data.climate_pipeline.schema.ClimatePipelineRequestArgs`
  (points + date range) is the manifest key, narrowed to a single point per
  write.
- :class:`~external_data.climate_pipeline.schema.Era5RequestArgs` is the ERA5
  *fetch spec* (variables), combined with the joined request's dates to build
  each per-point CDS call.

**Variables are a selection, not a fixture.** A request may name any non-empty
subset of the curated set and the ingest runs on it: the download is checked and
projected against what was *asked for*, and the curated columns left out are
written as nulls so the stored schema never changes. The selection is recorded on
the write's manifest key, which is what stops a subset write reading as coverage
for a later, wider request -- see :func:`variables_match`.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import shutil
import tempfile
import zipfile
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import cdsapi
import pandas as pd
import patito as pt
import polars as pl
import xarray as xr

import common.exceptions
from common.frames import BaseDataFrameSchema
from common.storage import columnar
from common.storage.manifest import ManifestRow
from external_data.climate_pipeline import failures, point_manifest, schema
from external_data.climate_pipeline.era5 import land_mask

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# CDS dataset id comes from the shared schema.
DATASET = schema.ERA5_SOURCE_DATASET

DATASET_NAME = "era5_weather_bronze"

DEFAULT_WRITER = "era5_land_bronze"

DEFAULT_BASE_DIR = "bronze"

_NETCDF_SUFFIXES = (".nc", ".nc4", ".netcdf")

KEY_COLUMNS = ["valid_time", "latitude", "longitude"]

# CDS publishes ERA5-Land on a 0.1 deg grid and snaps each request to it, so a
# node -- not the requested coordinate -- is the unit a download is keyed by.
NODE_DECIMALS = 1

# Short variable names CDS/xarray return, in the order bronze stores them --
# which is the order ``Era5LandBronzeSchema`` declares and the order already on
# disk, not the order ``schema.ERA5_VARIABLES`` asks for them in. Kept as its own
# list for that reason; ``test_value_columns_cover_the_curated_set`` is what stops
# the two drifting apart.
#
# This is the *stored* shape, not the fetched one: a request may ask for a subset
# (see :func:`selected_columns`), and the columns it leaves out are written as
# nulls. Bronze therefore has one schema whatever the selection -- which is what
# lets ``read_bronze_uris`` hand files from runs with different selections to a
# single ``pl.read_parquet``, and lets silver keep reading the columns it needs.
VALUE_COLUMNS = [
    "d2m",
    "t2m",
    "tp",
    "ssrd",
    "skt",
    "snowc",
    "sde",
    "stl1",
    "stl2",
    "stl3",
    "stl4",
    "u10",
    "v10",
]

# CDS snaps a request to the nearest 0.1 deg grid node but returns it with float
# noise (e.g. 37.4000000000011). Round the coordinates so the same node always
# resolves identically instead of scattering across float variants.
_LOCATION_DECIMALS = 4

# --------------------------------------------------------------------------- #
# Dataset schema (catalog / manifest)
# --------------------------------------------------------------------------- #


class Era5LandBronzeSchema(BaseDataFrameSchema):
    """
    One row per timestamp of the ERA5-Land point timeseries
    """

    valid_time: dt.datetime = pt.Field(
        dtype=pl.Datetime(time_unit="us", time_zone="UTC"),
        description="observation timestamp (UTC)",
    )
    latitude: float = pt.Field(dtype=pl.Float64, description="grid-node latitude")
    longitude: float = pt.Field(dtype=pl.Float64, description="grid-node longitude")

    d2m: float | None = pt.Field(
        dtype=pl.Float32, description="2m dewpoint temperature (K)"
    )
    t2m: float | None = pt.Field(dtype=pl.Float32, description="2m air temperature (K)")
    tp: float | None = pt.Field(dtype=pl.Float32, description="total precipitation (m)")
    ssrd: float | None = pt.Field(
        dtype=pl.Float32, description="surface solar radiation downwards (J/m^2)"
    )
    skt: float | None = pt.Field(dtype=pl.Float32, description="skin temperature (K)")
    snowc: float | None = pt.Field(dtype=pl.Float32, description="snow cover (%)")
    sde: float | None = pt.Field(
        dtype=pl.Float32, description="snow depth (m of water equivalent)"
    )
    stl1: float | None = pt.Field(
        dtype=pl.Float32, description="soil temperature level 1 (K)"
    )
    stl2: float | None = pt.Field(
        dtype=pl.Float32, description="soil temperature level 2 (K)"
    )
    stl3: float | None = pt.Field(
        dtype=pl.Float32, description="soil temperature level 3 (K)"
    )
    stl4: float | None = pt.Field(
        dtype=pl.Float32, description="soil temperature level 4 (K)"
    )
    u10: float | None = pt.Field(
        dtype=pl.Float32, description="10m U wind component (m/s)"
    )
    v10: float | None = pt.Field(
        dtype=pl.Float32, description="10m V wind component (m/s)"
    )


def selected_columns(era5: schema.Era5RequestArgs | None) -> list[str]:
    """
    The value columns a fetch for *era5* actually returns, in stored order.

    What the download is checked and projected against. The rest of
    :data:`VALUE_COLUMNS` is null in the write -- the stored schema is fixed, the
    fetched subset is not.
    """
    variables = schema.ERA5_VARIABLES if era5 is None else era5.variables
    selected = {schema.ERA5_VARIABLE_COLUMNS[v] for v in variables}
    return [column for column in VALUE_COLUMNS if column in selected]


def build_request(
    era5: schema.Era5RequestArgs,
    start_date: dt.date,
    end_date: dt.date,
    latitude: float,
    longitude: float,
) -> dict[str, Any]:
    """
    Build a CDS request payload for a single point: variables come from the ERA5
    fetch config, the date range from the joined request
    """
    return {
        "variable": list(era5.variables),
        "location": {"latitude": latitude, "longitude": longitude},
        "date": [f"{start_date:%Y-%m-%d}/{end_date:%Y-%m-%d}"],
        "data_format": "netcdf",
    }


def request_hash(request: dict[str, Any]) -> str:
    """
    Stable, order-independent short hash identifying a CDS request
    """
    payload = json.dumps(request, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Fetch
# --------------------------------------------------------------------------- #


def download_to_scratch(
    request: dict[str, Any],
    scratch_dir: Path,
    client: cdsapi.Client | None = None,
) -> Path:
    """
    Download the raw CDS archive into `scratch_dir` and return its path.

    The only place that sees the CDS response, so where failures are classified:
    cdsapi raises an untyped ``Exception``, and a rejected request will be rejected
    identically on retry. Re-raising as ``PermanentFetchError`` lets the flow
    record the point and move on rather than burn its retries.
    """
    client = client if client is not None else cdsapi.Client()

    target = scratch_dir / f"{request_hash(request)}.download"
    try:
        client.retrieve(DATASET, request, str(target))
    except Exception as exc:
        permanent = failures.as_permanent_cds(exc)
        if permanent is not None:
            raise permanent from exc
        raise
    return target


def extract_netcdfs(archive_path: Path, scratch_dir: Path) -> list[Path]:
    """
    Resolve a downloaded archive to the NetCDF file(s) inside `scratch_dir`
    """
    with open(archive_path, "rb") as handle:
        is_zip = handle.read(4)[:2] == b"PK"

    if not is_zip:
        return [archive_path]

    with zipfile.ZipFile(archive_path) as archive:
        members = sorted(
            name for name in archive.namelist() if name.endswith(_NETCDF_SUFFIXES)
        )
        if not members:
            raise ValueError(f"No NetCDF file found in {archive_path}")
        return [Path(archive.extract(name, scratch_dir)) for name in members]


# --------------------------------------------------------------------------- #
# Parse -> bronze table
# --------------------------------------------------------------------------- #


def netcdf_to_bronze_table(
    nc_paths: Path | Sequence[Path],
    request: dict[str, Any],
    expected: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Merge the per-group NetCDF file(s) into one wide table (one row per timestamp).

    *expected* is the value columns this download asked for -- the whole curated
    set by default. Checking against what was requested rather than against
    :data:`VALUE_COLUMNS` is what lets a subset fetch through: a variable nobody
    asked for is absent, not missing.
    """
    paths = [nc_paths] if isinstance(nc_paths, (str, Path)) else list(nc_paths)

    datasets = [xr.open_dataset(path, engine="h5netcdf") for path in paths]
    try:
        merged = xr.merge(datasets, compat="override", join="exact")
        df = merged.to_dataframe().reset_index()
    finally:
        for dataset in datasets:
            dataset.close()

    if "valid_time" not in df.columns and "time" in df.columns:
        df = df.rename(columns={"time": "valid_time"})

    # lat/lon are expected in the response; fail loud rather than silently
    # substituting the *requested* coordinate, which would land off the ERA5 grid
    # and make the partition / silver-join key request-dependent.
    missing_coords = [c for c in ("latitude", "longitude") if c not in df.columns]
    if missing_coords:
        location = request.get("location", {})
        msg = (
            f"ERA5-Land response is missing coordinate column(s) {missing_coords} "
            f"for requested point {location}; got columns {sorted(df.columns)}"
        )
        raise common.exceptions.PipelineValueError(msg)

    # Round the point coordinates to the grid precision
    df["latitude"] = df["latitude"].round(_LOCATION_DECIMALS)
    df["longitude"] = df["longitude"].round(_LOCATION_DECIMALS)

    # Fail loud if the download is missing a variable it *asked for*, rather than
    # silently writing a short table that only surfaces downstream
    wanted = list(VALUE_COLUMNS if expected is None else expected)
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        msg = (
            f"ERA5-Land download is missing expected variable(s) {missing}; "
            f"got columns {sorted(df.columns)}"
        )
        raise common.exceptions.PipelineValueError(msg)

    # Project to what was requested only. CDS/xarray can attach extra dims
    # so drop anything not explicitly ask for
    ordered = KEY_COLUMNS + wanted
    return df[ordered].sort_values("valid_time").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Write -> permanent storage
# --------------------------------------------------------------------------- #


def bronze_table_to_frame(table: pd.DataFrame) -> pl.DataFrame:
    """
    Convert the parsed pandas bronze table to a Polars frame, in the stored shape.

    A variable the fetch did not ask for is added as an all-null column rather
    than left out, so every write has the same columns whatever was selected.
    Two reasons that matters more than the few bytes it costs: ``read_bronze_uris``
    hands several points' files to one ``pl.read_parquet``, which needs them to
    agree on a schema; and ``Era5LandBronzeSchema`` -- validated on the way out --
    declares all of them. Which variables a write *holds* is recorded in its
    manifest key (``Era5PointSliceKey.variables``), not inferred from its columns.
    """
    frame = pl.from_pandas(table)
    present = [c for c in VALUE_COLUMNS if c in frame.columns]
    absent = [c for c in VALUE_COLUMNS if c not in frame.columns]
    return frame.with_columns(
        pl.col("valid_time").cast(pl.Datetime("us")).dt.replace_time_zone("UTC"),
        pl.col("latitude").round(_LOCATION_DECIMALS),
        pl.col("longitude").round(_LOCATION_DECIMALS),
        *[pl.col(c).cast(pl.Float32) for c in present],
        *[pl.lit(None, dtype=pl.Float32).alias(c) for c in absent],
    ).select(KEY_COLUMNS + VALUE_COLUMNS)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def _fetch_point_table(
    latitude: float,
    longitude: float,
    era5: schema.Era5RequestArgs,
    start_date: dt.date,
    end_date: dt.date,
    scratch_dir: Path,
    client: cdsapi.Client | None,
) -> pd.DataFrame:
    """
    Fetch and parse a single point's timeseries into a bronze table: variables
    from the ERA5 fetch config, date range from the joined request
    """
    request = build_request(era5, start_date, end_date, latitude, longitude)
    archive = download_to_scratch(request, scratch_dir, client=client)
    nc_paths = extract_netcdfs(archive, scratch_dir)
    return netcdf_to_bronze_table(nc_paths, request, selected_columns(era5))


def _require_values(table: pd.DataFrame, point: tuple[float, float]) -> None:
    """
    Reject a response carrying timestamps but no measurements.

    A point outside ERA5-Land's domain returns the full time axis with everything
    missing, which reads as success everywhere else: rows exist, the schema
    validates, the write lands. Storing it is worse than failing -- the entry would
    report the range as covered, so later runs skip the point and silver builds an
    all-null node with nothing saying why.

    *Permanent*, since the land mask does not change between runs. That hands the
    point to the existing machinery: recorded, not retried, dead-lettered, and
    skipped by silver.
    """
    present = [column for column in VALUE_COLUMNS if column in table.columns]
    if len(table) and any(table[column].notna().any() for column in present):
        return
    msg = (
        f"ERA5-Land returned no values for ({point[0]}, {point[1]}): "
        f"{len(present)} variable(s) missing across {len(table)} timestamp(s). "
        "ERA5-Land covers land only, so a point at sea -- or otherwise outside "
        "its domain -- returns timestamps with no data behind them."
    )
    raise failures.PermanentFetchError(msg)


def fetch_point_table(
    point: tuple[float, float],
    request: schema.ClimatePipelineRequestArgs,
    era5: schema.Era5RequestArgs | None = None,
    client: cdsapi.Client | None = None,
) -> pd.DataFrame:
    """
    Fetch one point's timeseries as a bronze table.

    The unit an orchestrator fans out over: one point, one CDS request, no shared
    state. It owns its scratch directory so concurrent callers cannot collide, and
    stays free of orchestration imports.

    Fails on an all-missing response rather than returning it -- see
    :func:`_require_values`.
    """
    era5 = era5 if era5 is not None else schema.Era5RequestArgs()
    latitude, longitude = point

    # Refuse a point with no land *before* paying for it: CDS does not error on
    # one, it queues, runs, and returns a full time axis of missing values.
    if land_mask.is_sea(point):
        msg = (
            f"ERA5-Land has no land at ({latitude}, {longitude}), so it serves no "
            "data there; refused without contacting CDS. ERA5-Land covers land "
            "only -- check the coordinate, or swap latitude and longitude if they "
            "are the wrong way round."
        )
        raise failures.PermanentFetchError(msg)

    scratch_dir = Path(tempfile.mkdtemp(prefix="era5_land_bronze_point_"))
    try:
        table = _fetch_point_table(
            latitude,
            longitude,
            era5,
            request.start_date,
            request.end_date,
            scratch_dir,
            client,
        )
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)
    _require_values(table, (latitude, longitude))
    return table


def node(point: Sequence[float]) -> tuple[float, float]:
    """
    The 0.1 deg grid node a point snaps to.

    What a fetch is really keyed by, so also the grain the manifest is looked up
    at -- reading back at finer precision would make a stored node invisible.
    """
    return (
        round(float(point[0]), NODE_DECIMALS),
        round(float(point[1]), NODE_DECIMALS),
    )


def unique_points(
    request: schema.ClimatePipelineRequestArgs,
) -> list[tuple[float, float]]:
    """
    The request's points, one per distinct ERA5 grid node, order preserved.

    CDS snaps to the nearest node, so points sharing one download identical data.
    De-duplicating here keeps the sequential and fanned-out paths in step.
    """
    seen: set[tuple[float, float]] = set()
    out: list[tuple[float, float]] = []
    for latitude, longitude in request.points:
        key = node((latitude, longitude))
        if key not in seen:
            seen.add(key)
            out.append((latitude, longitude))
    if len(out) < len(request.points):
        logger.info(
            "bronze: %d requested point(s) map to %d distinct grid node(s)",
            len(request.points),
            len(out),
        )
    return out


def point_params(
    point: tuple[float, float],
    request: schema.ClimatePipelineRequestArgs,
    era5: schema.Era5RequestArgs | None = None,
) -> schema.Era5PointSliceKey:
    """
    The manifest identity of one point's slice of a batch request.

    Names *the* point rather than a collection of one: the write covers a single
    node, which keeps the key readable and the readers free of a multi-point case.

    Carries the fetched ``variables`` too, since a write no longer implies the
    whole curated set -- the stored columns cannot tell the two apart, so the key
    has to. ``era5`` defaults to the curated set, which is what a default fetch
    holds.
    """
    return schema.Era5PointSliceKey(
        point=point_manifest.normalise(point),
        start_date=request.start_date,
        end_date=request.end_date,
        variables=schema.ERA5_VARIABLES if era5 is None else era5.variables,
    )


def write_point_bronze(
    table: pd.DataFrame,
    point: tuple[float, float],
    request: schema.ClimatePipelineRequestArgs,
    root_uri: str | Path,
    writer: str = DEFAULT_WRITER,
    write_time: dt.datetime | None = None,
    era5: schema.Era5RequestArgs | None = None,
) -> ManifestRow:
    """
    Write one point as its own single-file dataset.

    One file, not a partitioned directory: one point per write, so a location
    partition would be a directory holding one file. The manifest is the index, and
    every point owning an immutable write is what makes the fan-out safe with no
    storage changes.

    ``era5`` is the fetch config the *table* came from; it goes into the key so
    the entry records which variables it holds. Last and defaulted rather than
    beside ``request``: a caller that fetched the curated set -- every caller
    before subsets existed -- needs no change.
    """
    frame = bronze_table_to_frame(table)
    return columnar.write_dataset(
        frame,
        Era5LandBronzeSchema,
        DATASET_NAME,
        point_params(point, request, era5),
        str(root_uri),
        writer=writer,
        write_time=write_time,
    )


def stored_variables(row: ManifestRow) -> frozenset[str]:
    """
    The ERA5 variables a manifest entry holds values for.

    An entry written before the selection was recorded has no ``variables``, and
    is read as the full curated set -- correctly, since a subset could not be
    requested then.
    """
    params = json.loads(row.params_json)
    return frozenset(params.get("variables", schema.ERA5_VARIABLES))


def variables_match(
    era5: schema.Era5RequestArgs | None,
) -> Callable[[ManifestRow], bool]:
    """
    A :func:`point_manifest.covered_rows` / :func:`split_by_coverage` filter that
    only accepts an entry holding **every** variable the request asks for.

    Coverage is a superset test, not equality: an entry with the full curated set
    answers a three-variable request, so narrowing a selection re-fetches nothing.
    Widening one does re-fetch, which is the point -- without this filter the
    earlier, narrower write would read as coverage, the point would never be
    fetched again, and the added variables would stay null in bronze and in every
    silver built from it.

    The re-fetch supersedes the narrow write rather than merging with it: the new
    entry is the most recent for that node, so readers resolve to it.
    """
    wanted = frozenset(schema.ERA5_VARIABLES if era5 is None else era5.variables)

    def _matches(row: ManifestRow) -> bool:
        return wanted <= stored_variables(row)

    return _matches


def resolve_point_uris(
    request: schema.ClimatePipelineRequestArgs,
    root_uri: str | Path,
    as_of: dt.datetime | None = None,
    *,
    allow_missing: bool = False,
) -> dict[tuple[float, float], str]:
    """
    The parquet URI holding each requested grid node's bronze.

    Split from :func:`read_points_bronze` so a node-by-node caller resolves the
    whole request in **one** scan, then reads one node at a time. Resolving inside
    the loop would trade a memory problem for a latency one.
    """
    # Nodes, not exact points: neighbours sharing a node share one download, so
    # asking for both must not look like a half-covered request.
    rows, missing = point_manifest.covered_rows(
        request.points,
        DATASET_NAME,
        str(root_uri),
        point_manifest.read_point_key,
        request.start_date,
        request.end_date,
        grain=node,
        as_of=as_of,
    )
    if missing:
        detail = (
            f"{len(missing)} of {len(missing) + len(rows)} requested grid node(s) "
            f"for [{request.start_date}, {request.end_date}] under {root_uri}: "
            f"{point_manifest.describe_points(missing)}"
        )
        # Strict by default: returning whatever exists would build a silver table
        # quietly missing locations. ``allow_missing`` is for the caller that
        # already knows some points failed permanently.
        if not allow_missing or not rows:
            raise common.exceptions.PipelineValueError(
                f"no ERA5-Land bronze covering {detail}; ingest the bronze "
                "slice before building silver"
            )
        logger.warning(
            "ERA5-Land bronze is missing %s; continuing without them", detail
        )
    return {point: row.data_uri for point, row in rows.items()}


def read_bronze_uris(uris: Iterable[str]) -> pl.DataFrame:
    """Read already-resolved per-point bronze files into one deduped frame."""
    frame = pl.read_parquet(list(uris))
    return frame.unique(subset=KEY_COLUMNS).sort(KEY_COLUMNS)


def read_points_bronze(
    request: schema.ClimatePipelineRequestArgs,
    root_uri: str | Path,
    as_of: dt.datetime | None = None,
    *,
    allow_missing: bool = False,
) -> pl.DataFrame:
    """
    Read every requested point's bronze back as one frame.

    One scan plus one multi-file read, rather than a resolve and single-file read
    per point -- the loop is what gets slow at hundreds of points.
    """
    uris = resolve_point_uris(request, root_uri, as_of, allow_missing=allow_missing)
    return read_bronze_uris(uris.values())


def ingest_bronze(
    request: schema.ClimatePipelineRequestArgs,
    era5: schema.Era5RequestArgs | None = None,
    root_uri: str | Path = DEFAULT_BASE_DIR,
    writer: str = DEFAULT_WRITER,
    client: cdsapi.Client | None = None,
    write_time: dt.datetime | None = None,
) -> list[ManifestRow]:
    """
    Fetch the ERA5-Land timeseries for every point in ``request`` over its date
    range and write **one bronze dataset per grid node**, returning a manifest
    row for each. ``era5`` is the fetch config (variables); it defaults to the
    curated set, and any non-empty subset of it fetches just those -- the rest
    are stored as nulls and the selection is recorded on each entry.

    Per point, not per batch, so what this writes is what
    :func:`resolve_point_uris` (and therefore silver) can find -- a batch entry
    cannot answer "do I have this point?" and is invisible to every reader.

    Fetches the points **sequentially**. An orchestrator that wants them
    concurrent should map :func:`fetch_point_table` + :func:`write_point_bronze`
    itself -- the same two steps this function performs, which is why they are
    public.
    """
    if era5 is None:
        era5 = schema.Era5RequestArgs()

    if "://" not in str(root_uri):
        root_uri = str(Path(root_uri).resolve())

    points = unique_points(request)
    rows: list[ManifestRow] = []
    skipped: list[tuple[float, float]] = []
    for index, point in enumerate(points):
        try:
            table = fetch_point_table(point, request, era5, client)
        except failures.PermanentFetchError as exc:
            # Only *permanent* failures are absorbed. There is no retry layer on
            # this sequential path, so anything else may well succeed on another
            # attempt and must stay visible rather than be silently skipped.
            logger.warning(
                "bronze: point %d/%d (%.4f, %.4f) permanently unavailable, "
                "skipping: %s",
                index + 1,
                len(points),
                point[0],
                point[1],
                exc,
            )
            skipped.append(point)
            continue
        row = write_point_bronze(
            table,
            point,
            request,
            root_uri,
            writer=writer,
            write_time=write_time,
            era5=era5,
        )
        logger.info(
            "bronze: point %d/%d (%.4f, %.4f) -> %d row(s) -> %s",
            index + 1,
            len(points),
            point[0],
            point[1],
            len(table),
            row.data_uri,
        )
        rows.append(row)
    if not rows:
        msg = (
            f"no ERA5-Land bronze produced: all {len(points)} point(s) are "
            "permanently unavailable"
        )
        raise common.exceptions.PipelineValueError(msg)
    logger.info(
        "bronze: wrote %d node(s), one entry each; skipped %d permanently unavailable",
        len(rows),
        len(skipped),
    )
    return rows


if __name__ == "__main__":
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    _demo = schema.ClimatePipelineRequestArgs(
        points=((37.7749, -122.4194),),
        start_date=dt.date(2023, 6, 1),
        end_date=dt.date(2023, 6, 2),
    )
    for _row in ingest_bronze(_demo):
        print(_row.data_uri)
