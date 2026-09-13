"""
ERA5-Land x NSRDB -- silver join.

Reads the two bronze datasets (both keyed by the joined
:class:`~external_data.climate_pipeline.schema.ClimatePipelineRequestArgs`) and
joins them into one harmonized, hourly weather + solar table on the **ERA5-Land
0.1 deg grid**.

- ERA5-Land bronze -- hourly weather (temperatures in K, wind as u/v components,
  precip in m, snow), 0.1 deg grid. This is the *final* grid: every ERA5 node
  becomes a silver cell.
- NSRDB bronze -- 30/60-min solar/irradiance (W/m^2) plus ancillary met (relative
  humidity, surface pressure), GOES 4 km grid. Its finer cells are snapped onto
  the coarser ERA5 grid (nearest kept, others discarded).

The join is **ERA5-anchored** (left join on ``(node, hour)``): every ERA5
node/hour is kept; NSRDB fills in where a point snaps to it, else ``None``. This
module also owns the **NSRDB coverage clamp** -- deriving the NSRDB request from
the joined request and intersecting it with NSRDB's coverage window (a range
entirely outside coverage, e.g. all-2025, yields an ERA5-only grid).

**Built one grid node at a time**, each written as its own entry. Per-point
bronze makes the join independent per node, so a plain loop holds one node's data
at a time -- which is what keeps a thousands-of-points backfill (hundreds of
millions of rows) inside memory without streaming machinery. See
:func:`iter_silver_tables`.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from collections.abc import Iterator
from pathlib import Path

import patito as pt
import polars as pl

import common.exceptions
from common.frames import BaseDataFrameSchema
from common.storage import columnar
from common.storage.manifest import ManifestRow
from external_data.climate_pipeline import point_manifest, schema
from external_data.climate_pipeline.era5 import bronze as era5_bronze
from external_data.climate_pipeline.nsrdb import bronze as nsrdb_bronze

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DATASET_NAME = "era5_nsrdb_silver"

DEFAULT_WRITER = "era5_nsrdb_silver"

# Medallion roots (siblings): silver reads bronze from the bronze root and writes
# the silver dataset to the silver root.
DEFAULT_BRONZE_DIR = "bronze"
DEFAULT_SILVER_DIR = "silver"

# ERA5-Land grid step in degrees. NSRDB points snap to the nearest node by
# rounding their coordinates to this precision (round(coord, 1) == nearest node).
_GRID_DECIMALS = 1

# NSRDB columns carried into silver (mean-resampled to hourly). air_temperature /
# wind_speed are dropped in favor of ERA5; cloud_type / fill_flag are bronze QA.
_NSRDB_SOLAR_COLUMNS = [
    "ghi",
    "dni",
    "dhi",
    "clearsky_ghi",
    "clearsky_dni",
    "clearsky_dhi",
    "solar_zenith_angle",
    "surface_albedo",
]
# NSRDB ancillary met, under their raw bronze names (renamed on the way out).
_NSRDB_ANCILLARY_COLUMNS = ["relative_humidity", "surface_pressure"]
_NSRDB_MEAN_COLUMNS = _NSRDB_SOLAR_COLUMNS + _NSRDB_ANCILLARY_COLUMNS

# --------------------------------------------------------------------------- #
# Schema (catalog / manifest)
# --------------------------------------------------------------------------- #

# Row identity is the ERA5 0.1 deg node + hour; the grid is ERA5-anchored.
KEY_COLUMNS = ["era5_latitude", "era5_longitude", "valid_time"]
# No partition columns: one node per write, so partitioning would make a
# directory containing one file. The manifest is the index.

# Coordinate columns, in display order: the ERA5 grid node first, then the native
# NSRDB 4 km point that snapped onto it (null where no NSRDB covers the node/hour).
_COORD_COLUMNS = [
    "era5_latitude",
    "era5_longitude",
    "nsrdb_latitude",
    "nsrdb_longitude",
]
# ERA5-derived weather variables (appear before the NSRDB ones).
_WEATHER_COLUMNS = [
    "air_temperature_c",
    "dewpoint_c",
    "skin_temperature_c",
    "soil_temperature_c",
    "wind_speed_ms",
    "wind_direction_deg",
    "precipitation_mm",
    "snow_depth_m",
    "snow_cover_pct",
]
# NSRDB-derived ancillary met (grouped with the NSRDB solar variables, after ERA5).
_ANCILLARY_COLUMNS = ["relative_humidity_pct", "surface_pressure_mbar"]

# Column order: time, ERA5 node coords, NSRDB coords, all ERA5 (weather)
# variables, then all NSRDB (solar + ancillary) variables.
COLUMNS = (
    ["valid_time"]
    + _COORD_COLUMNS
    + _WEATHER_COLUMNS
    + _NSRDB_SOLAR_COLUMNS
    + _ANCILLARY_COLUMNS
)


class Era5NsrdbSilverSchema(BaseDataFrameSchema):
    """
    One row per ERA5 0.1 deg grid node x hour: ERA5 weather + NSRDB solar,
    harmonized. The grid is ERA5-anchored, so ``era5_*`` is always present;
    ``nsrdb_*`` and the solar/ancillary values are nullable (None where no NSRDB
    covers that node/hour -- e.g. a period past NSRDB's coverage such as 2025).
    """

    valid_time: dt.datetime = pt.Field(
        dtype=pl.Datetime(time_unit="us", time_zone="UTC"),
        description="interval timestamp (UTC, hourly)",
    )
    era5_latitude: float = pt.Field(
        dtype=pl.Float64, description="ERA5-Land 0.1 deg node latitude (grid key)"
    )
    era5_longitude: float = pt.Field(
        dtype=pl.Float64, description="ERA5-Land 0.1 deg node longitude (grid key)"
    )
    nsrdb_latitude: float | None = pt.Field(
        dtype=pl.Float64,
        description="native NSRDB GOES 4 km cell latitude that snapped onto this "
        "node (None where no NSRDB covers the node/hour)",
    )
    nsrdb_longitude: float | None = pt.Field(
        dtype=pl.Float64,
        description="native NSRDB GOES 4 km cell longitude that snapped onto this "
        "node (None where no NSRDB covers the node/hour)",
    )
    air_temperature_c: float | None = pt.Field(
        dtype=pl.Float32, description="2m air temperature (C; ERA5 t2m)"
    )
    dewpoint_c: float | None = pt.Field(
        dtype=pl.Float32, description="2m dewpoint temperature (C; ERA5 d2m)"
    )
    skin_temperature_c: float | None = pt.Field(
        dtype=pl.Float32, description="skin temperature (C; ERA5 skt)"
    )
    soil_temperature_c: float | None = pt.Field(
        dtype=pl.Float32, description="level-1 soil temperature (C; ERA5 stl1)"
    )
    wind_speed_ms: float | None = pt.Field(
        dtype=pl.Float32, description="10m wind speed (m/s; from ERA5 u10/v10)"
    )
    wind_direction_deg: float | None = pt.Field(
        dtype=pl.Float32,
        description="10m wind direction (deg FROM, met convention; from ERA5 u10/v10)",
    )
    precipitation_mm: float | None = pt.Field(
        dtype=pl.Float32, description="total precipitation (mm; ERA5 tp x1000)"
    )
    snow_depth_m: float | None = pt.Field(
        dtype=pl.Float32, description="snow depth (m water equivalent; ERA5 sde)"
    )
    snow_cover_pct: float | None = pt.Field(
        dtype=pl.Float32, description="snow cover (%; ERA5 snowc)"
    )
    ghi: float | None = pt.Field(
        dtype=pl.Float32,
        description="global horizontal irradiance (W/m^2, hourly mean)",
    )
    dni: float | None = pt.Field(
        dtype=pl.Float32, description="direct normal irradiance (W/m^2, hourly mean)"
    )
    dhi: float | None = pt.Field(
        dtype=pl.Float32,
        description="diffuse horizontal irradiance (W/m^2, hourly mean)",
    )
    clearsky_ghi: float | None = pt.Field(
        dtype=pl.Float32, description="clear-sky GHI (W/m^2, hourly mean)"
    )
    clearsky_dni: float | None = pt.Field(
        dtype=pl.Float32, description="clear-sky DNI (W/m^2, hourly mean)"
    )
    clearsky_dhi: float | None = pt.Field(
        dtype=pl.Float32, description="clear-sky DHI (W/m^2, hourly mean)"
    )
    solar_zenith_angle: float | None = pt.Field(
        dtype=pl.Float32, description="solar zenith angle (degrees, hourly mean)"
    )
    surface_albedo: float | None = pt.Field(
        dtype=pl.Float32, description="surface albedo (fraction, hourly mean)"
    )
    relative_humidity_pct: float | None = pt.Field(
        dtype=pl.Float32, description="relative humidity (%; NSRDB, hourly mean)"
    )
    surface_pressure_mbar: float | None = pt.Field(
        dtype=pl.Float32, description="surface pressure (mbar; NSRDB, hourly mean)"
    )


# --------------------------------------------------------------------------- #
# Source-request derivation (owns the NSRDB coverage clamp)
# --------------------------------------------------------------------------- #


def nsrdb_request(
    request: schema.ClimatePipelineRequestArgs,
) -> schema.ClimatePipelineRequestArgs | None:
    """
    Derive the NSRDB read/ingest request from the joined request by intersecting
    its date range with NSRDB's coverage window (:data:`schema.NSRDB_VALID_YEARS`).

    Returns ``None`` when the range lies entirely outside coverage (no solar to
    fetch/read -> an ERA5-only silver grid). ERA5 always uses the full request.
    """
    start = max(request.start_date, schema.NSRDB_MIN_DATE)
    end = min(request.end_date, schema.NSRDB_MAX_DATE)
    if start > end:
        logger.warning(
            "requested range %s..%s is entirely outside NSRDB coverage (%s..%s); "
            "building ERA5-only silver (None solar)",
            request.start_date,
            request.end_date,
            schema.NSRDB_MIN_DATE,
            schema.NSRDB_MAX_DATE,
        )
        return None
    if (start, end) != (request.start_date, request.end_date):
        logger.info(
            "NSRDB window clamped to %s..%s (request %s..%s)",
            start,
            end,
            request.start_date,
            request.end_date,
        )
    return schema.ClimatePipelineRequestArgs(
        points=request.points, start_date=start, end_date=end
    )


# --------------------------------------------------------------------------- #
# Transform helpers
# --------------------------------------------------------------------------- #


def nsrdb_to_hourly(frame: pl.DataFrame) -> pl.DataFrame:
    """
    Resample the (possibly multi-point) NSRDB frame to hourly means (irradiance
    is a flux), one row per point x hour, carrying each cell's coords as provenance
    """
    return (
        frame.with_columns(pl.col("valid_time").dt.truncate("1h").alias("valid_time"))
        .group_by(["valid_time", "latitude", "longitude"])
        .agg(*[pl.col(c).mean() for c in _NSRDB_MEAN_COLUMNS])
        .rename({"latitude": "nsrdb_latitude", "longitude": "nsrdb_longitude"})
    )


def _snap_to_nearest_node(hourly_solar: pl.DataFrame) -> pl.DataFrame:
    """
    Map each NSRDB point onto its nearest ERA5 0.1 deg node and, where several
    points snap to the same node, keep only the one closest to the node center
    (discarding the rest -- the finer 4 km grid collapses onto the coarser grid).

    Returns the hourly solar restricted to the winning points, with the snapped
    ``node_latitude``/``node_longitude`` attached as the join key.
    """
    # Rounding a coordinate to the grid step yields its nearest node exactly.
    with_node = hourly_solar.with_columns(
        pl.col("nsrdb_latitude").round(_GRID_DECIMALS).alias("node_latitude"),
        pl.col("nsrdb_longitude").round(_GRID_DECIMALS).alias("node_longitude"),
    )

    # Pick one winning point per node from the distinct points (not per hour), so
    # the whole time series of the nearest point is kept. Squared Euclidean
    # distance to the node center suffices within a single cell; ties break
    # deterministically on the point coordinates.
    winners = (
        with_node.select(
            "nsrdb_latitude", "nsrdb_longitude", "node_latitude", "node_longitude"
        )
        .unique()
        .with_columns(
            (
                (pl.col("nsrdb_latitude") - pl.col("node_latitude")) ** 2
                + (pl.col("nsrdb_longitude") - pl.col("node_longitude")) ** 2
            ).alias("_dist2")
        )
        .sort(
            [
                "node_latitude",
                "node_longitude",
                "_dist2",
                "nsrdb_latitude",
                "nsrdb_longitude",
            ]
        )
        .group_by(["node_latitude", "node_longitude"], maintain_order=True)
        .first()
        .select("nsrdb_latitude", "nsrdb_longitude")
    )

    return with_node.join(
        winners, on=["nsrdb_latitude", "nsrdb_longitude"], how="inner"
    )


def era5_features(frame: pl.DataFrame) -> pl.DataFrame:
    """
    Convert ERA5-Land bronze to friendly units and derive wind speed/direction
    """
    # meteorological wind direction: the compass bearing the wind blows FROM
    bearing = pl.arctan2(pl.col("v10"), pl.col("u10")) * (180.0 / math.pi)
    return frame.select(
        "valid_time",
        pl.col("latitude").alias("era5_latitude"),
        pl.col("longitude").alias("era5_longitude"),
        (pl.col("t2m") - 273.15).alias("air_temperature_c"),
        (pl.col("d2m") - 273.15).alias("dewpoint_c"),
        (pl.col("skt") - 273.15).alias("skin_temperature_c"),
        (pl.col("stl1") - 273.15).alias("soil_temperature_c"),
        ((pl.col("u10") ** 2 + pl.col("v10") ** 2).sqrt()).alias("wind_speed_ms"),
        ((270.0 - bearing) % 360.0).alias("wind_direction_deg"),
        (pl.col("tp") * 1000.0).alias("precipitation_mm"),
        pl.col("sde").alias("snow_depth_m"),
        pl.col("snowc").alias("snow_cover_pct"),
    )


def _rename_ancillary(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.rename(
        {
            "relative_humidity": "relative_humidity_pct",
            "surface_pressure": "surface_pressure_mbar",
        }
    )


def _null_solar(weather: pl.DataFrame) -> pl.DataFrame:
    """
    Attach all NSRDB-sourced columns as nulls -- used when no NSRDB covers the
    grid (an ERA5-only build), so the frame still matches the silver schema.
    """
    return weather.with_columns(
        pl.lit(None, dtype=pl.Float64).alias("nsrdb_latitude"),
        pl.lit(None, dtype=pl.Float64).alias("nsrdb_longitude"),
        *[pl.lit(None, dtype=pl.Float32).alias(c) for c in _NSRDB_SOLAR_COLUMNS],
        *[pl.lit(None, dtype=pl.Float32).alias(c) for c in _ANCILLARY_COLUMNS],
    )


def join_grid(
    era5_frame: pl.DataFrame,
    nsrdb_frame: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """
    Combine the ERA5 weather grid with NSRDB solar into the silver table.

    The grid is **ERA5-anchored**: every ERA5 node/hour is kept, and each NSRDB
    point is snapped onto its nearest ERA5 0.1 deg node (nearest kept) and joined
    on ``(node, hour)`` as a **left** join. So:

    - ERA5 nodes/hours with no NSRDB (a 2025 window past NSRDB coverage, or a node
      no NSRDB point snapped to) keep the weather with ``None`` solar;
    - NSRDB points snapping to a node absent from the ERA5 slice are dropped.

    When ``nsrdb_frame`` is ``None`` or empty (no NSRDB at all), the result is an
    ERA5-only grid with every solar/ancillary column ``None``.
    """
    weather = era5_features(era5_frame).with_columns(
        pl.col("era5_latitude").round(_GRID_DECIMALS).alias("node_latitude"),
        pl.col("era5_longitude").round(_GRID_DECIMALS).alias("node_longitude"),
    )

    if nsrdb_frame is None or nsrdb_frame.is_empty():
        combined = _null_solar(weather)
    else:
        solar = _rename_ancillary(_snap_to_nearest_node(nsrdb_to_hourly(nsrdb_frame)))
        combined = weather.join(
            solar,
            on=["node_latitude", "node_longitude", "valid_time"],
            how="left",
        )

    if combined.is_empty():
        msg = "ERA5 slice is empty; nothing to build"
        raise common.exceptions.PipelineValueError(msg)
    return _shape_to_schema(combined)


def _shape_to_schema(frame: pl.DataFrame) -> pl.DataFrame:
    """
    Cast to the silver schema dtypes, order/sort, and validate (schema is the
    single source of dtype truth)
    """
    shaped = (
        Era5NsrdbSilverSchema.DataFrame(frame.select(COLUMNS)).cast().sort(KEY_COLUMNS)
    )
    Era5NsrdbSilverSchema.validate(shaped)
    return shaped


# --------------------------------------------------------------------------- #
# Read bronze -> silver table
# --------------------------------------------------------------------------- #


def _resolve_root_uri(root_uri: str | Path) -> str:
    if "://" not in str(root_uri):
        return str(Path(root_uri).resolve())
    return str(root_uri)


Span = tuple[dt.date, dt.date]


def _extend(span: Span | None, frame: pl.DataFrame) -> Span | None:
    """Widen a running (lo, hi) date span to include *frame*'s timestamps."""
    if frame.is_empty():
        return span
    lo = frame["valid_time"].min().date()  # ty:ignore[unresolved-attribute]
    hi = frame["valid_time"].max().date()  # ty:ignore[unresolved-attribute]
    if span is None:
        return (lo, hi)
    return (min(span[0], lo), max(span[1], hi))


def _log_coverage(
    label: str,
    span: Span | None,
    requested_start: dt.date,
    requested_end: dt.date,
) -> None:
    """
    Report a source's *actual* coverage and warn about any requested span it
    misses. Gaps are not fatal -- the join fills them.

    Takes an accumulated span, not a frame, so the build reports once rather than
    per node; at a thousand nodes the per-node version buries the warning.
    """
    if span is None:
        logger.warning(
            "%s bronze is empty for %s..%s", label, requested_start, requested_end
        )
        return
    actual_lo, actual_hi = span
    logger.info("%s bronze coverage: %s..%s", label, actual_lo, actual_hi)
    if requested_start < actual_lo:
        logger.warning(
            "%s does not cover %s..%s (starts at %s); those hours get None",
            label,
            requested_start,
            actual_lo,
            actual_lo,
        )
    if requested_end > actual_hi:
        logger.warning(
            "%s does not cover %s..%s (ends at %s); those hours get None",
            label,
            actual_hi,
            requested_end,
            actual_hi,
        )


def _node_groups(
    era5_uris: dict[point_manifest.Point, str],
    nsrdb_uris: dict[point_manifest.Point, str],
) -> dict[point_manifest.Point, tuple[str, list[str]]]:
    """
    One build unit per ERA5 node: its bronze file, plus the NSRDB files whose
    points snap onto it.

    Grouping by **node**, not by requested point, keeps the units disjoint: two
    points in one cell share a node, and splitting them would emit the same
    ``(node, hour)`` rows twice.

    NSRDB points snapping onto a node with no ERA5 bronze are dropped, exactly as
    the ERA5-anchored left join drops them.
    """
    groups: dict[point_manifest.Point, tuple[str, list[str]]] = {
        node: (uri, []) for node, uri in era5_uris.items()
    }
    for point, uri in nsrdb_uris.items():
        group = groups.get(era5_bronze.node(point))
        if group is not None:
            group[1].append(uri)
    return groups


def iter_silver_tables(
    request: schema.ClimatePipelineRequestArgs,
    bronze_root: str | Path,
    as_of: dt.datetime | None = None,
    *,
    allow_missing_points: bool = False,
) -> Iterator[tuple[point_manifest.Point, pl.DataFrame]]:
    """
    Yield ``(node, silver table)`` one ERA5 grid node at a time.

    This is the memory bound: thousands of points over ~27 years is hundreds of
    millions of rows, and an eager whole-request join exhausts a normal machine.
    Per-point bronze makes the join independent per node, so looping keeps memory
    flat however many points were requested. Each iteration calls
    :func:`join_grid` unchanged, so a node's rows match a whole-request build.

    Both manifests resolve up front, one scan each: the loop only touches parquet,
    and the strict/``allow_missing_points`` decision is made once.

    ``as_of`` bounds the reads. ``allow_missing_points`` builds from whatever
    bronze exists rather than refusing on a gap -- off by default, since the usual
    cause is a bronze step never run, and a table quietly covering fewer locations
    is worse than an error.
    """
    bronze_root = _resolve_root_uri(bronze_root)
    # Bronze is written one entry per point, so gather the requested points from
    # the manifest rather than resolving a single batch entry.
    era5_uris = era5_bronze.resolve_point_uris(
        request, bronze_root, as_of, allow_missing=allow_missing_points
    )

    nsrdb_req = nsrdb_request(request)
    nsrdb_uris: dict[point_manifest.Point, str] = {}
    if nsrdb_req is not None:
        # ERA5-Land is hourly, so join the 60-minute NSRDB slice (interval is part
        # of the NSRDB dataset identity; a 30-minute fetch is a distinct dataset).
        nsrdb_uris = nsrdb_bronze.resolve_point_uris(
            nsrdb_req,
            schema.NsrdbRequestArgs(interval=schema.NSRDB_SILVER_INTERVAL),
            bronze_root,
            as_of,
            allow_missing=allow_missing_points,
        )

    groups = _node_groups(era5_uris, nsrdb_uris)
    era5_span: Span | None = None
    nsrdb_span: Span | None = None

    # Sorted so the concatenated tables are already in KEY_COLUMNS order.
    for node in sorted(groups):
        era5_uri, point_uris = groups[node]
        era5_frame = era5_bronze.read_bronze_uris([era5_uri])
        era5_span = _extend(era5_span, era5_frame)
        nsrdb_frame = None
        if point_uris:
            nsrdb_frame = nsrdb_bronze.read_bronze_uris(point_uris)
            nsrdb_span = _extend(nsrdb_span, nsrdb_frame)
        yield node, join_grid(era5_frame, nsrdb_frame)

    _log_coverage("ERA5-Land", era5_span, request.start_date, request.end_date)
    if nsrdb_req is not None:
        _log_coverage(
            "NSRDB (60-minute)",
            nsrdb_span,
            nsrdb_req.start_date,
            nsrdb_req.end_date,
        )


def build_silver_table(
    request: schema.ClimatePipelineRequestArgs,
    bronze_root: str | Path,
    as_of: dt.datetime | None = None,
    *,
    allow_missing_points: bool = False,
) -> pl.DataFrame:
    """
    The whole request's silver grid as one frame.

    A convenience for callers wanting every node at once -- tests and ad-hoc work.
    It **defeats the memory bound** by construction, so the pipeline uses
    :func:`ingest_silver` and backfills should iterate
    :func:`iter_silver_tables`. Arguments as on that function.
    """
    tables = [
        table
        for _node, table in iter_silver_tables(
            request, bronze_root, as_of, allow_missing_points=allow_missing_points
        )
    ]
    if not tables:
        # Unreachable via the resolvers, but an empty concat fails obscurely.
        msg = "no ERA5-Land bronze resolved; nothing to build"
        raise common.exceptions.PipelineValueError(msg)
    return pl.concat(tables)


# --------------------------------------------------------------------------- #
# Write + orchestration
# --------------------------------------------------------------------------- #


def node_params(
    node: point_manifest.Point, request: schema.ClimatePipelineRequestArgs
) -> schema.PointSliceKey:
    """
    The manifest identity of one grid node's silver table.

    Keyed by the *node*, since that is what a silver row is keyed by -- two
    requested points in one cell resolve to the entry they share.
    """
    return schema.PointSliceKey(
        point=point_manifest.normalise(node),
        start_date=request.start_date,
        end_date=request.end_date,
    )


def write_silver(
    table: pl.DataFrame,
    node: point_manifest.Point,
    request: schema.ClimatePipelineRequestArgs,
    silver_root: str | Path,
    writer: str = DEFAULT_WRITER,
    write_time: dt.datetime | None = None,
) -> ManifestRow:
    """
    Write one grid node's silver table as its own single-file dataset.

    One file, not a partitioned directory: the table holds one node, so a location
    partition would be a directory containing one file. The manifest is the index,
    and each node owning an immutable write is what lets nodes be rebuilt
    independently.
    """
    return columnar.write_dataset(
        table,
        Era5NsrdbSilverSchema,
        DATASET_NAME,
        node_params(node, request),
        _resolve_root_uri(silver_root),
        writer=writer,
        write_time=write_time,
    )


def read_points_silver(
    request: schema.ClimatePipelineRequestArgs,
    root_uri: str | Path,
    as_of: dt.datetime | None = None,
    *,
    allow_missing: bool = False,
) -> pl.DataFrame:
    """
    Read the silver tables for a request's grid nodes back as one frame.

    One entry per node, so there is no batch key to resolve -- the manifest is the
    index, read at the grain :func:`write_silver` writes at.
    """
    rows, missing = point_manifest.covered_rows(
        request.points,
        DATASET_NAME,
        str(root_uri),
        point_manifest.read_point_key,
        request.start_date,
        request.end_date,
        grain=era5_bronze.node,
        as_of=as_of,
    )
    if missing:
        detail = (
            f"{len(missing)} of {len(missing) + len(rows)} requested grid node(s) "
            f"for [{request.start_date}, {request.end_date}] under {root_uri}: "
            f"{point_manifest.describe_points(missing)}"
        )
        if not allow_missing or not rows:
            raise common.exceptions.PipelineValueError(
                f"no {DATASET_NAME} covering {detail}; build the silver table first"
            )
        logger.warning(
            "%s is missing %s; continuing without them", DATASET_NAME, detail
        )
    frame = pl.read_parquet([row.data_uri for row in rows.values()])
    return frame.unique(subset=KEY_COLUMNS).sort(KEY_COLUMNS)


def ingest_silver(
    request: schema.ClimatePipelineRequestArgs,
    bronze_root: str | Path = DEFAULT_BRONZE_DIR,
    silver_root: str | Path = DEFAULT_SILVER_DIR,
    writer: str = DEFAULT_WRITER,
    write_time: dt.datetime | None = None,
    as_of: dt.datetime | None = None,
    allow_missing_points: bool = False,
) -> list[ManifestRow]:
    """
    Full silver step: read the bronze slices for ``request`` from ``bronze_root``,
    join them into the harmonized hourly grid, and write the result to
    ``silver_root`` as **one entry per ERA5 grid node**.

    Each node is written as it is built, so nothing accumulates -- this is the
    path that keeps a large backfill inside memory. Returns every row written,
    since lineage has to name all of them.

    ``as_of`` and ``allow_missing_points`` are as on :func:`iter_silver_tables`.
    """
    bronze_root = _resolve_root_uri(bronze_root)
    silver_root = _resolve_root_uri(silver_root)

    rows: list[ManifestRow] = []
    total_rows = 0
    for node, table in iter_silver_tables(
        request, bronze_root, as_of, allow_missing_points=allow_missing_points
    ):
        rows.append(
            write_silver(
                table,
                node,
                request,
                silver_root=silver_root,
                writer=writer,
                write_time=write_time,
            )
        )
        total_rows += table.height

    logger.info(
        "silver: wrote %s -> %d grid node dataset(s), %d row(s)",
        DATASET_NAME,
        len(rows),
        total_rows,
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
    for silver_row in ingest_silver(_demo):
        print(silver_row.data_uri)
