"""
Tests for the climate_pipeline ERA5 x NSRDB silver join.

Mirrors tests/era5_nsrdb_silver, adapted to the schema-wired module: one
``ClimatePipelineRequestArgs`` (points + dates) keys everything, and the NSRDB
coverage clamp lives in ``silver.nsrdb_request``. Uses the committed multi-node
golden fixtures (ERA5 bronze A/B/C + a 2025 window; NSRDB points P1-P4).
"""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from common.exceptions import PipelineValueError
from external_data.climate_pipeline import schema
from external_data.climate_pipeline import silver
from external_data.climate_pipeline.era5 import bronze as era5_bronze
from external_data.climate_pipeline.nsrdb import bronze as nsrdb_bronze

FIXTURES = Path(__file__).parent / "fixtures"
ERA5_SAMPLE = FIXTURES / "era5_bronze_sample.parquet"
NSRDB_SAMPLE = FIXTURES / "nsrdb_bronze_sample.parquet"
SILVER_GOLDEN = FIXTURES / "era5_nsrdb_silver_golden.parquet"

_UTC = dt.timezone.utc
_FIXED_WRITE_TIME = dt.datetime(2026, 7, 21, tzinfo=_UTC)

# The grid the fixtures were built for:
#   ERA5 nodes A/B/C = (37.8,-122.4)/(37.7,-122.4)/(37.9,-122.4)
#   NSRDB P1/P2 -> A (P1 nearer), P3 -> B, P4 -> D (37.6,-122.4, orphan)
_NODE_A = (37.8, -122.4)
_NODE_B = (37.7, -122.4)
_NODE_C = (37.9, -122.4)
_NODE_D_ORPHAN = (37.6, -122.4)
_P1_LAT = 37.77  # nearest point to node A -> kept
_P2_LAT = 37.83  # also snaps to A -> discarded

_ERA5_POINTS = (_NODE_A, _NODE_B, _NODE_C)
_START = dt.date(2023, 6, 1)
_END = dt.date(2025, 6, 1)  # runs past NSRDB coverage -> None solar in 2025
_EXPECTED_ROWS = 21  # A=9 (6x2023 + 3x2025), B=6, C=6; orphan D dropped

_HAS_NSRDB = bool(__import__("os").environ.get("NSRDB_API_KEY")) or (
    nsrdb_bronze._rc_path().exists()
)
_HAS_CDS = (Path.home() / ".cdsapirc").exists()
_HAS_CREDENTIALS = _HAS_NSRDB and _HAS_CDS


def _hour(year: int, hour: int) -> dt.datetime:
    return dt.datetime(year, 6, 1, hour, tzinfo=_UTC)


@pytest.fixture(scope="session")
def era5_frame() -> pl.DataFrame:
    return pl.read_parquet(ERA5_SAMPLE)


@pytest.fixture(scope="session")
def nsrdb_frame() -> pl.DataFrame:
    return pl.read_parquet(NSRDB_SAMPLE)


@pytest.fixture(scope="session")
def silver_golden() -> pl.DataFrame:
    return pl.read_parquet(SILVER_GOLDEN)


def _request() -> schema.ClimatePipelineRequestArgs:
    return schema.ClimatePipelineRequestArgs(
        points=_ERA5_POINTS, start_date=_START, end_date=_END
    )


# --------------------------------------------------------------------------- #
# NSRDB coverage clamp (derivation)
# --------------------------------------------------------------------------- #


def test_nsrdb_request_clamps_to_coverage() -> None:
    nreq = silver.nsrdb_request(_request())
    assert nreq is not None
    assert nreq.points == _ERA5_POINTS
    assert nreq.start_date == _START
    assert nreq.end_date == schema.NSRDB_MAX_DATE  # 2025 tail clamped off


def test_nsrdb_request_none_when_entirely_outside() -> None:
    beyond = schema.NSRDB_VALID_YEARS[-1] + 1
    req = schema.ClimatePipelineRequestArgs(
        points=(_NODE_A,),
        start_date=dt.date(beyond, 1, 1),
        end_date=dt.date(beyond, 12, 31),
    )
    assert silver.nsrdb_request(req) is None


# --------------------------------------------------------------------------- #
# Pure transforms
# --------------------------------------------------------------------------- #


def test_nsrdb_to_hourly_means_per_point(nsrdb_frame: pl.DataFrame) -> None:
    hourly = silver.nsrdb_to_hourly(nsrdb_frame)
    # 4 points x 6 hours (00:00..05:00) of half-hourly source -> 24 rows
    assert hourly.height == 24
    hour = _hour(2023, 2)
    got = hourly.filter(
        (pl.col("nsrdb_latitude") == _P1_LAT) & (pl.col("valid_time") == hour)
    )["ghi"][0]
    expected = nsrdb_frame.filter(
        (pl.col("latitude") == _P1_LAT)
        & (pl.col("valid_time").dt.truncate("1h") == hour)
    )["ghi"].mean()
    assert got == pytest.approx(expected, abs=1e-3)


def test_era5_features_units_and_wind(era5_frame: pl.DataFrame) -> None:
    feats = silver.era5_features(era5_frame).sort("valid_time")
    src = era5_frame.sort("valid_time")
    f0, s0 = feats.row(0, named=True), src.row(0, named=True)
    assert f0["air_temperature_c"] == pytest.approx(s0["t2m"] - 273.15, abs=1e-2)
    assert f0["precipitation_mm"] == pytest.approx(s0["tp"] * 1000.0, abs=1e-2)
    assert f0["wind_speed_ms"] == pytest.approx(
        math.hypot(s0["u10"], s0["v10"]), abs=1e-3
    )
    expected_dir = (270.0 - math.degrees(math.atan2(s0["v10"], s0["u10"]))) % 360.0
    assert f0["wind_direction_deg"] == pytest.approx(expected_dir, abs=1e-2)


def test_snap_keeps_nearest_and_discards_rest(nsrdb_frame: pl.DataFrame) -> None:
    snapped = silver._snap_to_nearest_node(silver.nsrdb_to_hourly(nsrdb_frame))
    node_a = snapped.filter(
        (pl.col("node_latitude") == _NODE_A[0])
        & (pl.col("node_longitude") == _NODE_A[1])
    )
    assert node_a["nsrdb_latitude"].unique().to_list() == [_P1_LAT]
    assert _P2_LAT not in snapped["nsrdb_latitude"].to_list()


# --------------------------------------------------------------------------- #
# Grid join
# --------------------------------------------------------------------------- #


def test_join_grid_matches_golden(
    era5_frame: pl.DataFrame,
    nsrdb_frame: pl.DataFrame,
    silver_golden: pl.DataFrame,
) -> None:
    table = silver.join_grid(era5_frame, nsrdb_frame)
    assert table.columns == silver.COLUMNS
    silver.Era5NsrdbSilverSchema.validate(table)
    assert_frame_equal(
        table.sort(silver.KEY_COLUMNS), silver_golden.sort(silver.KEY_COLUMNS)
    )


def test_grid_grain_and_key_is_the_era5_node(silver_golden: pl.DataFrame) -> None:
    nodes = silver_golden.select(["era5_latitude", "era5_longitude"]).unique()
    assert set(nodes.rows()) == {_NODE_A, _NODE_B, _NODE_C}
    assert silver_golden.columns[0] == "valid_time"
    assert silver_golden.columns[1:5] == [
        "era5_latitude",
        "era5_longitude",
        "nsrdb_latitude",
        "nsrdb_longitude",
    ]
    assert (
        silver_golden.select(
            ["era5_latitude", "era5_longitude", "valid_time"]
        ).n_unique()
        == silver_golden.height
    )


def test_era5_variables_ordered_before_nsrdb(silver_golden: pl.DataFrame) -> None:
    cols = silver_golden.columns
    last_weather = max(cols.index(c) for c in silver._WEATHER_COLUMNS)
    first_nsrdb = min(
        cols.index(c) for c in silver._NSRDB_SOLAR_COLUMNS + silver._ANCILLARY_COLUMNS
    )
    assert last_weather < first_nsrdb


def test_era5_only_nodes_have_none_solar(silver_golden: pl.DataFrame) -> None:
    node_c = silver_golden.filter(pl.col("era5_latitude") == _NODE_C[0])
    assert node_c.height > 0
    assert node_c["air_temperature_c"].is_not_null().all()
    assert node_c["ghi"].is_null().all()
    assert node_c["nsrdb_latitude"].is_null().all()

    a_2025 = silver_golden.filter(
        (pl.col("era5_latitude") == _NODE_A[0])
        & (pl.col("valid_time").dt.year() == 2025)
    )
    assert a_2025.height == 3
    assert a_2025["air_temperature_c"].is_not_null().all()
    assert a_2025["ghi"].is_null().all()


def test_orphan_nsrdb_node_is_dropped(silver_golden: pl.DataFrame) -> None:
    assert _NODE_D_ORPHAN[0] not in silver_golden["era5_latitude"].to_list()
    assert silver_golden["era5_latitude"].is_not_null().all()


def test_join_grid_era5_only_none_solar(era5_frame: pl.DataFrame) -> None:
    table = silver.join_grid(era5_frame, None)
    assert table.columns == silver.COLUMNS
    silver.Era5NsrdbSilverSchema.validate(table)
    assert table.height == era5_frame.height
    assert table["air_temperature_c"].is_not_null().all()
    for col in ("ghi", "dni", "nsrdb_latitude", "relative_humidity_pct"):
        assert table[col].is_null().all()


# --------------------------------------------------------------------------- #
# End-to-end (bronze written to a temp lakehouse, one entry per point)
# --------------------------------------------------------------------------- #


def _rows_for(frame: pl.DataFrame, point: tuple[float, float]) -> pl.DataFrame:
    """The seed frame's rows for one point, matched at the grid-node grain."""
    node = era5_bronze.node(point)
    return frame.filter(
        (pl.col("latitude").round(era5_bronze.NODE_DECIMALS) == node[0])
        & (pl.col("longitude").round(era5_bronze.NODE_DECIMALS) == node[1])
    )


def _seed_bronze(
    bronze_root: str,
    request: schema.ClimatePipelineRequestArgs,
    era5_frame: pl.DataFrame,
    nsrdb_frame: pl.DataFrame | None,
) -> None:
    """
    Seed bronze the way the flow writes it: one manifest entry per point.

    Deliberately mirrors production rather than writing one batch entry -- a
    batch key is not readable per point, so seeding one would test a layout the
    pipeline no longer produces.
    """
    for point in era5_bronze.unique_points(request):
        era5_bronze.write_point_bronze(
            _rows_for(era5_frame, point).to_pandas(),
            point,
            request,
            bronze_root,
            writer="seed",
            write_time=_FIXED_WRITE_TIME,
        )
    if nsrdb_frame is not None:
        nreq = silver.nsrdb_request(request)
        assert nreq is not None
        nsrdb_args = schema.NsrdbRequestArgs(interval=schema.NSRDB_SILVER_INTERVAL)
        # request_units expands to point-years; the write unit is the point, with
        # its years already combined, so collapse back to distinct points.
        points = dict.fromkeys(
            point for point, _year in nsrdb_bronze.request_units(nreq)
        )
        for point in points:
            nsrdb_bronze.write_point_bronze(
                _rows_for(nsrdb_frame, point),
                point,
                nreq,
                nsrdb_args,
                bronze_root,
                writer="seed",
                write_time=_FIXED_WRITE_TIME,
            )


def test_ingest_silver_round_trips(
    era5_frame: pl.DataFrame,
    nsrdb_frame: pl.DataFrame,
    silver_golden: pl.DataFrame,
    tmp_path: Path,
) -> None:
    bronze_root = str(tmp_path / "bronze")
    silver_root = str(tmp_path / "silver")
    request = _request()
    _seed_bronze(bronze_root, request, era5_frame, nsrdb_frame)

    rows = silver.ingest_silver(
        request,
        bronze_root=bronze_root,
        silver_root=silver_root,
        write_time=_FIXED_WRITE_TIME,
    )
    # One entry per grid node, not one per request: the table is built node by
    # node so a large backfill never holds the whole grid in memory.
    assert len(rows) == len(_ERA5_POINTS)
    assert {row.dataset_name for row in rows} == {silver.DATASET_NAME}

    # silver lands under the silver root...
    assert list(Path(silver_root).rglob(f"{silver.DATASET_NAME}/**/*.parquet"))
    # ...and not under the bronze root.
    assert not list(Path(bronze_root).rglob(f"{silver.DATASET_NAME}/**/*.parquet"))

    back = silver.read_points_silver(request, silver_root)
    assert_frame_equal(
        back.sort(silver.KEY_COLUMNS).select(silver_golden.columns),
        silver_golden.sort(silver.KEY_COLUMNS),
        check_row_order=True,
    )


def test_ingest_fails_loud_when_nsrdb_missing(
    era5_frame: pl.DataFrame, tmp_path: Path
) -> None:
    # In-coverage request, but only ERA5 bronze seeded -> NSRDB read fails loud.
    bronze_root = str(tmp_path / "bronze")
    silver_root = str(tmp_path / "silver")
    request = _request()
    # ERA5 seeded, NSRDB deliberately absent, so the NSRDB read is what fails.
    _seed_bronze(bronze_root, request, era5_frame, None)
    with pytest.raises(PipelineValueError, match="NSRDB bronze covering"):
        silver.ingest_silver(request, bronze_root=bronze_root, silver_root=silver_root)


def test_ingest_era5_only_when_no_nsrdb_coverage(
    era5_frame: pl.DataFrame, tmp_path: Path
) -> None:
    # A range past NSRDB coverage -> nsrdb_request None -> ERA5-only silver, no
    # NSRDB read. Derived from the coverage window so a coverage bump can't
    # silently turn this into a different test.
    beyond = schema.NSRDB_MAX_DATE.year + 1
    bronze_root = str(tmp_path / "bronze")
    silver_root = str(tmp_path / "silver")
    request = schema.ClimatePipelineRequestArgs(
        points=_ERA5_POINTS,
        start_date=dt.date(beyond, 1, 1),
        end_date=dt.date(beyond, 12, 31),
    )
    _seed_bronze(bronze_root, request, era5_frame, None)
    silver.ingest_silver(request, bronze_root=bronze_root, silver_root=silver_root)

    back = silver.read_points_silver(request, silver_root)
    assert back.height > 0
    assert back["ghi"].is_null().all()
    assert back["air_temperature_c"].is_not_null().all()


# --------------------------------------------------------------------------- #
# Live integration (both APIs: CDS + NSRDB)
# --------------------------------------------------------------------------- #


@pytest.mark.integration
@pytest.mark.skipif(
    not _HAS_CREDENTIALS, reason="needs NSRDB + CDS credentials for the live APIs"
)
def test_ingest_live_reconstructs_grid(tmp_path: Path) -> None:
    """
    Full pipeline against the real APIs: ingest a small ERA5 + NSRDB grid over an
    overlapping 2023 window, then join into the silver grid.
    """
    bronze_root = str(tmp_path / "bronze")
    silver_root = str(tmp_path / "silver")
    request = schema.ClimatePipelineRequestArgs(
        points=((37.8, -122.4), (37.7, -122.4)),
        start_date=dt.date(2023, 6, 1),
        end_date=dt.date(2023, 6, 2),
    )

    era5_bronze.ingest_bronze(request, root_uri=bronze_root)
    nreq = silver.nsrdb_request(request)
    assert nreq is not None
    nsrdb_bronze.ingest_bronze(nreq, root_uri=bronze_root)
    silver.ingest_silver(request, bronze_root=bronze_root, silver_root=silver_root)

    back = silver.read_points_silver(request, silver_root)
    silver.Era5NsrdbSilverSchema.validate(back)
    assert back.height > 0
    assert back["era5_latitude"].n_unique() >= 2
    lit = back.filter(pl.col("ghi").is_not_null())
    assert (lit["ghi"] >= 0).all()
    assert (lit["ghi"] > 0).any()
    weather = back.filter(pl.col("air_temperature_c").is_not_null())
    assert (weather["air_temperature_c"] > -20).all()
    assert (weather["air_temperature_c"] < 45).all()


# --------------------------------------------------------------------------- #
# Building over a gap the provider will never fill
# --------------------------------------------------------------------------- #


def test_silver_refuses_a_gap_by_default(
    era5_frame: pl.DataFrame, nsrdb_frame: pl.DataFrame, tmp_path: Path
) -> None:
    # The usual cause of a gap is a bronze step that was never run, so a table
    # quietly covering fewer locations than asked for is worse than an error.
    bronze_root = str(tmp_path / "bronze")
    request = _request()
    _seed_bronze(bronze_root, request, era5_frame, nsrdb_frame)
    wider = schema.ClimatePipelineRequestArgs(
        points=(*_ERA5_POINTS, (34.05, -118.24)),  # a node never ingested
        start_date=_START,
        end_date=_END,
    )
    with pytest.raises(PipelineValueError, match="1 of 4 requested grid node"):
        silver.build_silver_table(wider, bronze_root)


def test_silver_builds_over_a_gap_when_allowed(
    era5_frame: pl.DataFrame, nsrdb_frame: pl.DataFrame, tmp_path: Path
) -> None:
    # For the point a provider permanently cannot serve: one dead location must
    # not block every other one.
    bronze_root = str(tmp_path / "bronze")
    request = _request()
    _seed_bronze(bronze_root, request, era5_frame, nsrdb_frame)
    wider = schema.ClimatePipelineRequestArgs(
        points=(*_ERA5_POINTS, (34.05, -118.24)),
        start_date=_START,
        end_date=_END,
    )

    table = silver.build_silver_table(wider, bronze_root, allow_missing_points=True)

    assert table.height == _EXPECTED_ROWS  # identical to the gap-free build
    nodes = set(table.select(["era5_latitude", "era5_longitude"]).unique().rows())
    assert nodes == {_NODE_A, _NODE_B, _NODE_C}  # the missing node is simply absent


def test_silver_refuses_when_everything_is_missing_even_if_allowed(
    tmp_path: Path,
) -> None:
    # Permissive is not "build from nothing": an empty silver table is a failed
    # run wearing a success badge.
    with pytest.raises(PipelineValueError):
        silver.build_silver_table(
            _request(), str(tmp_path / "empty"), allow_missing_points=True
        )


# --------------------------------------------------------------------------- #
# Building node by node (the memory bound)
# --------------------------------------------------------------------------- #


def test_iter_yields_one_node_at_a_time(
    era5_frame: pl.DataFrame, nsrdb_frame: pl.DataFrame, tmp_path: Path
) -> None:
    # The memory bound is exactly this: whatever was requested, an iteration
    # holds one node's rows, never the whole grid.
    bronze_root = str(tmp_path / "bronze")
    request = _request()
    _seed_bronze(bronze_root, request, era5_frame, nsrdb_frame)

    yielded = list(silver.iter_silver_tables(request, bronze_root))

    assert [node for node, _ in yielded] == sorted([_NODE_A, _NODE_B, _NODE_C])
    for node, table in yielded:
        nodes = set(table.select(["era5_latitude", "era5_longitude"]).unique().rows())
        assert nodes == {node}


def test_node_by_node_build_matches_the_whole_grid_join(
    era5_frame: pl.DataFrame,
    nsrdb_frame: pl.DataFrame,
    silver_golden: pl.DataFrame,
    tmp_path: Path,
) -> None:
    # The whole point of looping is that it costs nothing in correctness: the
    # concatenated per-node tables are the table the one-shot join produced.
    bronze_root = str(tmp_path / "bronze")
    request = _request()
    _seed_bronze(bronze_root, request, era5_frame, nsrdb_frame)

    table = silver.build_silver_table(request, bronze_root)

    assert table.columns == silver.COLUMNS
    silver.Era5NsrdbSilverSchema.validate(table)
    assert_frame_equal(
        table.sort(silver.KEY_COLUMNS),
        silver_golden.sort(silver.KEY_COLUMNS),
        check_row_order=True,
    )


def test_points_sharing_a_cell_write_one_entry(
    era5_frame: pl.DataFrame, nsrdb_frame: pl.DataFrame, tmp_path: Path
) -> None:
    # Grouping by node rather than by requested point is what stops this: two
    # points in one 0.1 deg cell built separately would emit the same
    # (node, hour) rows twice, once per unit.
    bronze_root = str(tmp_path / "bronze")
    silver_root = str(tmp_path / "silver")
    request = schema.ClimatePipelineRequestArgs(
        points=((_P1_LAT, -122.4), (_P2_LAT, -122.4)),  # both snap to node A
        start_date=_START,
        end_date=_END,
    )
    _seed_bronze(bronze_root, request, era5_frame, nsrdb_frame)

    rows = silver.ingest_silver(
        request, bronze_root=bronze_root, silver_root=silver_root
    )

    assert len(rows) == 1
    back = silver.read_points_silver(request, silver_root)
    assert set(back.select(["era5_latitude", "era5_longitude"]).unique().rows()) == {
        _NODE_A
    }
    assert back.select(silver.KEY_COLUMNS).n_unique() == back.height
    # ...and the nearer of the two points is still the one that wins the cell.
    assert back["nsrdb_latitude"].drop_nulls().unique().to_list() == [_P1_LAT]


def test_each_node_is_its_own_entry_readable_on_its_own(
    era5_frame: pl.DataFrame, nsrdb_frame: pl.DataFrame, tmp_path: Path
) -> None:
    # Per-node entries are what let a later run rebuild one node without
    # touching the rest, so each has to resolve independently.
    bronze_root = str(tmp_path / "bronze")
    silver_root = str(tmp_path / "silver")
    request = _request()
    _seed_bronze(bronze_root, request, era5_frame, nsrdb_frame)
    silver.ingest_silver(request, bronze_root=bronze_root, silver_root=silver_root)

    one_node = schema.ClimatePipelineRequestArgs(
        points=(_NODE_B,), start_date=_START, end_date=_END
    )
    back = silver.read_points_silver(one_node, silver_root)
    assert set(back.select(["era5_latitude", "era5_longitude"]).unique().rows()) == {
        _NODE_B
    }


def test_read_points_silver_refuses_a_node_never_built(
    era5_frame: pl.DataFrame, nsrdb_frame: pl.DataFrame, tmp_path: Path
) -> None:
    # Strict by default on the read side too: silently returning the nodes that
    # happen to exist is how a partial grid reaches a caller unnoticed.
    bronze_root = str(tmp_path / "bronze")
    silver_root = str(tmp_path / "silver")
    request = _request()
    _seed_bronze(bronze_root, request, era5_frame, nsrdb_frame)
    silver.ingest_silver(request, bronze_root=bronze_root, silver_root=silver_root)

    wider = schema.ClimatePipelineRequestArgs(
        points=(*_ERA5_POINTS, (34.05, -118.24)), start_date=_START, end_date=_END
    )
    with pytest.raises(PipelineValueError, match="1 of 4 requested grid node"):
        silver.read_points_silver(wider, silver_root)

    partial = silver.read_points_silver(wider, silver_root, allow_missing=True)
    assert set(partial.select(["era5_latitude", "era5_longitude"]).unique().rows()) == {
        _NODE_A,
        _NODE_B,
        _NODE_C,
    }
