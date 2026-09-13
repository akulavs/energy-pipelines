"""
Tests for the domain half of the data_fetch climate helper.

Covers consolidation, the coverage pre-check, the read-back, and the coordinate
parsing the command line uses. The orchestration -- deciding whether to run the
pipeline at all -- lives in ``batch_jobs.data_fetch.climate`` and is tested with
the other flow-side code, since it needs Prefect.

Silver is the only dataset the helper plans, reads or returns; the bronze it is
joined from is still seeded here, because that is what the silver is built from.

Reuses the committed climate_pipeline silver fixtures (ERA5 nodes A/B/C, NSRDB
points P1-P4) rather than minting new ones.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from external_data.climate_pipeline import point_manifest, schema, silver
from external_data.climate_pipeline.era5 import bronze as era5_bronze
from external_data.climate_pipeline.nsrdb import bronze as nsrdb_bronze
from external_data.data_fetch import climate, consolidate

FIXTURES = Path(__file__).parents[1] / "climate_pipeline" / "silver" / "fixtures"
ERA5_SAMPLE = FIXTURES / "era5_bronze_sample.parquet"
NSRDB_SAMPLE = FIXTURES / "nsrdb_bronze_sample.parquet"
SILVER_GOLDEN = FIXTURES / "era5_nsrdb_silver_golden.parquet"

_UTC = dt.timezone.utc
_FIXED_WRITE_TIME = dt.datetime(2026, 7, 21, tzinfo=_UTC)

# The grid the fixtures were built for.
_NODE_A = (37.8, -122.4)
_NODE_B = (37.7, -122.4)
_NODE_C = (37.9, -122.4)
_ERA5_POINTS = (_NODE_A, _NODE_B, _NODE_C)

# Two NSRDB points inside node A. P1 is nearer the node centre, so the silver
# snap keeps it and discards P2.
_P1 = (37.77, -122.42)
_P2 = (37.83, -122.44)

_START = dt.date(2023, 6, 1)
_END = dt.date(2025, 6, 1)


@pytest.fixture(scope="session")
def era5_frame() -> pl.DataFrame:
    return pl.read_parquet(ERA5_SAMPLE)


@pytest.fixture(scope="session")
def nsrdb_frame() -> pl.DataFrame:
    return pl.read_parquet(NSRDB_SAMPLE)


@pytest.fixture(scope="session")
def silver_golden() -> pl.DataFrame:
    return pl.read_parquet(SILVER_GOLDEN)


def _rows_for_node(frame: pl.DataFrame, point: tuple[float, float]) -> pl.DataFrame:
    """Every fixture row whose coordinates fall in *point*'s ERA5 node."""
    node = era5_bronze.node(point)
    return frame.filter(
        (pl.col("latitude").round(era5_bronze.NODE_DECIMALS) == node[0])
        & (pl.col("longitude").round(era5_bronze.NODE_DECIMALS) == node[1])
    )


def _request(points=_ERA5_POINTS, start=_START, end=_END):
    return schema.ClimatePipelineRequestArgs(
        points=points, start_date=start, end_date=end
    )


def _seed_bronze(
    root: str,
    request: schema.ClimatePipelineRequestArgs,
    era5_frame: pl.DataFrame,
    nsrdb_frame: pl.DataFrame | None = None,
    era5_points: tuple[tuple[float, float], ...] | None = None,
) -> None:
    """Seed bronze the way the pipeline writes it: one entry per point."""
    points = (
        era5_points if era5_points is not None else era5_bronze.unique_points(request)
    )
    for point in points:
        era5_bronze.write_point_bronze(
            _rows_for_node(era5_frame, point).to_pandas(),
            point,
            request,
            root,
            writer="seed",
            write_time=_FIXED_WRITE_TIME,
        )
    if nsrdb_frame is None:
        return
    nreq = silver.nsrdb_request(request)
    assert nreq is not None
    args = schema.NsrdbRequestArgs(interval=schema.NSRDB_SILVER_INTERVAL)
    for point in nreq.points:
        nsrdb_bronze.write_point_bronze(
            _rows_for_node(nsrdb_frame, point),
            point,
            nreq,
            args,
            root,
            writer="seed",
            write_time=_FIXED_WRITE_TIME,
        )


def _build_silver(root: str, request: schema.ClimatePipelineRequestArgs) -> None:
    silver.ingest_silver(
        request,
        bronze_root=root,
        silver_root=root,
        writer="seed",
        write_time=_FIXED_WRITE_TIME,
    )


def _store(root: str, era5_frame, nsrdb_frame, request=None) -> None:
    """A store as a completed run would leave it: bronze written, silver built."""
    request = request if request is not None else _request()
    _seed_bronze(root, request, era5_frame, nsrdb_frame)
    _build_silver(root, request)


def _datasets(root: str, points, start=_START, end=_END) -> climate.ClimateDatasets:
    grid = consolidate.consolidate(points)
    plan = climate.plan_coverage(grid, start, end, root)
    return climate.ClimateDatasets(
        silver=climate.read_silver(grid, start, end, root),
        points=grid,
        coverage=plan,
        ran_pipeline=False,
    )


# --------------------------------------------------------------------------- #
# Consolidation
# --------------------------------------------------------------------------- #


def test_consolidate_collapses_points_sharing_a_node() -> None:
    grid = consolidate.consolidate([_P1, _P2])
    # One node, so one call per source -- NSRDB included, since silver keeps only
    # one NSRDB point per node anyway.
    assert grid.nodes == (_NODE_A,)
    assert grid.node_by_input == {_P1: _NODE_A, _P2: _NODE_A}


def test_consolidate_drops_duplicate_coordinates() -> None:
    grid = consolidate.consolidate([_P1, _P1, _NODE_B])
    assert grid.nodes == (_NODE_A, _NODE_B)


def test_consolidate_normalises_before_comparing() -> None:
    # Beyond POINT_DECIMALS the two coordinates are the same point, and the
    # manifest compares them at that precision.
    grid = consolidate.consolidate([(37.1234567, -122.4), (37.1234568, -122.4)])
    assert len(grid.nodes) == 1


def test_consolidate_rejects_an_empty_request() -> None:
    with pytest.raises(ValueError, match="at least one point"):
        consolidate.consolidate([])


# --------------------------------------------------------------------------- #
# Coverage pre-check
# --------------------------------------------------------------------------- #


def test_plan_coverage_on_an_empty_store_has_nothing(tmp_path: Path) -> None:
    grid = consolidate.consolidate(_ERA5_POINTS)
    plan = climate.plan_coverage(grid, _START, _END, str(tmp_path / "empty"))

    assert not plan.complete
    assert plan.silver.covered == ()
    assert set(plan.silver.missing) == set(_ERA5_POINTS)


def test_plan_coverage_ignores_bronze_without_silver(
    era5_frame: pl.DataFrame, nsrdb_frame: pl.DataFrame, tmp_path: Path
) -> None:
    """
    Bronze alone is not coverage. Silver is what the caller gets, so a store with
    every bronze slice and no silver still has work to do.
    """
    root = str(tmp_path / "climate")
    _seed_bronze(root, _request(), era5_frame, nsrdb_frame)

    plan = climate.plan_coverage(
        consolidate.consolidate(_ERA5_POINTS), _START, _END, root
    )
    assert not plan.complete


def test_plan_coverage_is_complete_once_silver_exists(
    era5_frame: pl.DataFrame, nsrdb_frame: pl.DataFrame, tmp_path: Path
) -> None:
    root = str(tmp_path / "climate")
    _store(root, era5_frame, nsrdb_frame)

    plan = climate.plan_coverage(
        consolidate.consolidate(_ERA5_POINTS), _START, _END, root
    )

    # This is the property the whole pre-check exists for: nothing to run.
    assert plan.complete
    assert set(plan.silver.covered) == set(_ERA5_POINTS)


def test_plan_coverage_ignores_a_narrower_write(
    era5_frame: pl.DataFrame, nsrdb_frame: pl.DataFrame, tmp_path: Path
) -> None:
    root = str(tmp_path / "climate")
    # Built for two days only, then asked about the full range.
    _store(root, era5_frame, nsrdb_frame, _request(end=dt.date(2023, 6, 2)))

    plan = climate.plan_coverage(
        consolidate.consolidate(_ERA5_POINTS), _START, _END, root
    )

    # Present is not covering: a write falling short of the request is a gap.
    assert set(plan.silver.missing) == set(_ERA5_POINTS)


def test_plan_coverage_has_no_nsrdb_window_outside_coverage(tmp_path: Path) -> None:
    beyond = schema.NSRDB_MAX_DATE.year + 1
    plan = climate.plan_coverage(
        consolidate.consolidate((_NODE_A,)),
        dt.date(beyond, 1, 1),
        dt.date(beyond, 12, 31),
        str(tmp_path / "empty"),
    )

    # No solar in range: the silver grid carries null solar for those hours, and
    # saying so is what makes those nulls expected rather than a gap.
    assert plan.nsrdb_range is None


def test_recorded_gaps_is_empty_without_a_flow_record(tmp_path: Path) -> None:
    grid = consolidate.consolidate(_ERA5_POINTS)
    # Only the pipeline flow writes those records, so a store no flow has touched
    # explains no gaps.
    assert climate.recorded_gaps(grid, str(tmp_path / "empty")) == {}


# --------------------------------------------------------------------------- #
# Reading the silver grid back
# --------------------------------------------------------------------------- #


def test_read_silver_returns_the_pipeline_grid(
    era5_frame: pl.DataFrame,
    nsrdb_frame: pl.DataFrame,
    silver_golden: pl.DataFrame,
    tmp_path: Path,
) -> None:
    root = str(tmp_path / "climate")
    _store(root, era5_frame, nsrdb_frame)

    grid = consolidate.consolidate(_ERA5_POINTS)
    back = climate.read_silver(grid, _START, _END, root)

    # The same grid the silver join builds when the pipeline runs it.
    assert_frame_equal(
        back.sort(silver.KEY_COLUMNS).select(silver_golden.columns),
        silver_golden.sort(silver.KEY_COLUMNS),
        check_row_order=True,
    )


def test_read_silver_is_era5_only_outside_nsrdb_coverage(
    era5_frame: pl.DataFrame, tmp_path: Path
) -> None:
    beyond = schema.NSRDB_MAX_DATE.year + 1
    root = str(tmp_path / "climate")
    start, end = dt.date(beyond, 1, 1), dt.date(beyond, 12, 31)
    request = _request(points=(_NODE_A,), start=start, end=end)
    _seed_bronze(root, request, era5_frame)
    _build_silver(root, request)

    grid = consolidate.consolidate((_NODE_A,))
    back = climate.read_silver(grid, start, end, root)

    # Null solar, not missing rows: no NSRDB covers that window.
    assert back["ghi"].is_null().all()
    assert back["air_temperature_c"].is_not_null().all()


def test_read_silver_fails_loud_on_an_unexplained_gap(
    era5_frame: pl.DataFrame, nsrdb_frame: pl.DataFrame, tmp_path: Path
) -> None:
    root = str(tmp_path / "climate")
    # Node C never built: a gap nothing recorded, which should stop a read.
    _store(root, era5_frame, nsrdb_frame, _request(points=(_NODE_A, _NODE_B)))

    grid = consolidate.consolidate(_ERA5_POINTS)
    with pytest.raises(Exception, match=silver.DATASET_NAME):
        climate.read_silver(grid, _START, _END, root)


def test_read_silver_returns_exactly_what_is_on_disk(
    era5_frame: pl.DataFrame, nsrdb_frame: pl.DataFrame, tmp_path: Path
) -> None:
    """
    The frame holds the parquet's own rows, not a transformation of them. Only
    the concat, de-dupe, trim and sort the reader documents may differ.
    """
    root = str(tmp_path / "climate")
    _store(root, era5_frame, nsrdb_frame)

    grid = consolidate.consolidate(_ERA5_POINTS)
    back = climate.read_silver(grid, _START, _END, root)

    rows, _missing = point_manifest.covered_rows(
        grid.nodes,
        silver.DATASET_NAME,
        root,
        point_manifest.read_point_key,
        _START,
        _END,
        grain=era5_bronze.node,
    )
    disk = pl.read_parquet([row.data_uri for row in rows.values()])
    assert_frame_equal(
        back,
        disk.unique(subset=silver.KEY_COLUMNS).sort(silver.KEY_COLUMNS),
        check_column_order=False,
    )


def test_read_silver_validates_against_the_declared_schema(
    era5_frame: pl.DataFrame, nsrdb_frame: pl.DataFrame, tmp_path: Path
) -> None:
    """
    Worth asserting rather than assuming: ``read_dataset`` validates on the way
    in, but the wider-write fallback reads the parquet directly and does not, so
    a schema change would surface here first.
    """
    root = str(tmp_path / "climate")
    _store(root, era5_frame, nsrdb_frame)

    grid = consolidate.consolidate(_ERA5_POINTS)
    silver.Era5NsrdbSilverSchema.validate(climate.read_silver(grid, _START, _END, root))


def test_a_wider_stored_write_still_reads_and_validates(
    era5_frame: pl.DataFrame, nsrdb_frame: pl.DataFrame, tmp_path: Path
) -> None:
    """
    The fallback path: a stored write spanning more than the request.

    ``read_dataset`` matches ``params_json`` exactly, so it cannot resolve a
    wider write -- the reader falls back to the manifest row the planner already
    found. That path skips schema validation, so this asserts it explicitly.
    """
    root = str(tmp_path / "climate")
    _store(root, era5_frame, nsrdb_frame)
    narrow_end = dt.date(2023, 6, 2)

    grid = consolidate.consolidate(_ERA5_POINTS)
    plan = climate.plan_coverage(grid, _START, narrow_end, root)
    assert plan.complete, "a wider write should cover a narrower ask"

    back = climate.read_silver(grid, _START, narrow_end, root)
    assert back.height > 0
    silver.Era5NsrdbSilverSchema.validate(back)


def test_read_silver_trims_a_wider_write_to_the_requested_range(
    era5_frame: pl.DataFrame, nsrdb_frame: pl.DataFrame, tmp_path: Path
) -> None:
    """
    Coverage is containment, so a stored write routinely spans more than the
    request -- and the rows handed back must still be the days asked for.

    Without the trim a request for two days of a two-year write comes back with
    both years, which reads as data the caller never asked for and quietly
    changes any aggregate computed over it.
    """
    root = str(tmp_path / "climate")
    _store(root, era5_frame, nsrdb_frame)
    narrow_end = dt.date(2023, 6, 2)

    grid = consolidate.consolidate(_ERA5_POINTS)
    back = climate.read_silver(grid, _START, narrow_end, root)

    assert back.height > 0
    outside = back.filter(
        (pl.col("valid_time").dt.date() < _START)
        | (pl.col("valid_time").dt.date() > narrow_end)
    )
    assert outside.is_empty()


def test_two_coordinates_in_one_node_collapse_to_one_result(
    era5_frame: pl.DataFrame, nsrdb_frame: pl.DataFrame, tmp_path: Path
) -> None:
    root = str(tmp_path / "climate")
    _store(root, era5_frame, nsrdb_frame)

    grid = consolidate.consolidate([(37.7712, -122.4233), (37.7749, -122.4194)])
    back = climate.read_silver(grid, _START, _END, root)

    # One node, so one set of rows -- not the same hours twice.
    assert back.select("era5_latitude", "era5_longitude").unique().height == 1


def test_dataset_files_names_the_parquet_behind_the_grid(
    era5_frame: pl.DataFrame, nsrdb_frame: pl.DataFrame, tmp_path: Path
) -> None:
    root = str(tmp_path / "climate")
    _store(root, era5_frame, nsrdb_frame)
    result = _datasets(root, _ERA5_POINTS)

    files = climate.dataset_files(result, _START, _END, root)

    # One file per grid node, and each one actually there.
    assert len(files) == len(_ERA5_POINTS)
    assert all(Path(uri).exists() for uri in files)


# --------------------------------------------------------------------------- #
# Getting at one requested coordinate's rows
# --------------------------------------------------------------------------- #


def test_rows_for_takes_the_coordinate_that_was_requested(
    era5_frame: pl.DataFrame, nsrdb_frame: pl.DataFrame, tmp_path: Path
) -> None:
    """
    The frame is keyed by the node, never by the coordinate a caller typed, so
    slicing it by hand means knowing the snapping rule.
    """
    root = str(tmp_path / "climate")
    _store(root, era5_frame, nsrdb_frame)

    nudged = (37.7712, -122.4233)
    assert era5_bronze.node(nudged) == _NODE_A
    rows = _datasets(root, [nudged]).rows_for(nudged)

    assert rows.height > 0
    assert rows.select("era5_latitude", "era5_longitude").unique().rows() == [_NODE_A]


def test_rows_for_rejects_a_coordinate_the_request_did_not_cover(
    era5_frame: pl.DataFrame, nsrdb_frame: pl.DataFrame, tmp_path: Path
) -> None:
    root = str(tmp_path / "climate")
    _store(root, era5_frame, nsrdb_frame)
    result = _datasets(root, [_NODE_A])

    # Names what the request did cover, rather than raising a bare KeyError.
    with pytest.raises(Exception, match="was not part of this request"):
        result.rows_for((40.0, -74.0))


def test_rows_for_can_attach_the_node(
    era5_frame: pl.DataFrame, nsrdb_frame: pl.DataFrame, tmp_path: Path
) -> None:
    root = str(tmp_path / "climate")
    _store(root, era5_frame, nsrdb_frame)
    result = _datasets(root, [_NODE_A])

    rows = result.rows_for(_NODE_A, with_node=True)
    assert rows.select("node_latitude", "node_longitude").unique().rows() == [_NODE_A]


def test_rows_for_leaves_the_frame_alone_by_default(
    era5_frame: pl.DataFrame, nsrdb_frame: pl.DataFrame, tmp_path: Path
) -> None:
    """
    The node columns stay opt-in because they cost two properties worth keeping:
    a frame carrying them fails its schema validation, and no longer matches the
    parquet it was read from.
    """
    root = str(tmp_path / "climate")
    _store(root, era5_frame, nsrdb_frame)
    result = _datasets(root, [_NODE_A])

    plain = result.rows_for(_NODE_A)
    assert "node_latitude" not in plain.columns
    silver.Era5NsrdbSilverSchema.validate(plain)

    with pytest.raises(Exception):
        silver.Era5NsrdbSilverSchema.validate(result.rows_for(_NODE_A, with_node=True))


# --------------------------------------------------------------------------- #
# The point mapping
# --------------------------------------------------------------------------- #


def test_nsrdb_cell_comes_from_silver(
    era5_frame: pl.DataFrame, nsrdb_frame: pl.DataFrame, tmp_path: Path
) -> None:
    """
    Node A holds two NSRDB points in the fixtures, and silver keeps only the one
    nearer the node centre. The mapping has to name that one -- naming the
    discarded point would send a caller to rows the join never used.
    """
    root = str(tmp_path / "climate")
    _store(root, era5_frame, nsrdb_frame)
    result = _datasets(root, [_NODE_A])

    kept = result.nsrdb_cell_by_node()[_NODE_A]
    joined = result.silver["nsrdb_latitude"].drop_nulls().unique().to_list()
    assert joined == [kept[0]]
    # P1 is the nearer of the two fixture points, so that is the one kept.
    assert kept[0] == _P1[0]


def test_describe_mapping_groups_the_points_that_share_a_node(
    era5_frame: pl.DataFrame, nsrdb_frame: pl.DataFrame, tmp_path: Path
) -> None:
    """
    Grouped so a consolidation reads as one fact -- "these two are the same
    call" -- rather than as separate lines a reader has to notice share a node.
    """
    root = str(tmp_path / "climate")
    _store(root, era5_frame, nsrdb_frame)

    result = _datasets(root, [(37.7712, -122.4233), (37.7749, -122.4194), _NODE_B])
    lines = result.describe_mapping()

    # One line per node, not per requested coordinate.
    assert len(lines) == 2
    assert any(
        "(37.7712, -122.4233), (37.7749, -122.4194) -> ERA5/silver (37.8, -122.4)"
        in line
        for line in lines
    )
    # And the NSRDB cell is named, which is the key the join drew on.
    assert any("NSRDB (" in line for line in lines)


def test_the_mapping_does_not_truncate_a_coordinate() -> None:
    """
    ``%g`` counts significant digits, so it renders -122.4233 as -122.423 -- a
    different place, and a key that indexes nothing. The mapping exists to name
    exact keys, so every digit has to survive.
    """
    assert climate._coord(-122.4233) == "-122.4233"
    assert climate._coord(37.7712) == "37.7712"
    # A whole number keeps its decimal: "-122" reads like a different place.
    assert climate._coord(-122.0) == "-122.0"
    # And no trailing padding from the fixed-precision render.
    assert climate._coord(37.29) == "37.29"


# --------------------------------------------------------------------------- #
# Coordinate input
# --------------------------------------------------------------------------- #


def test_parse_point_reads_a_lat_lon_pair() -> None:
    assert climate.parse_point("37.7749,-122.4194") == (37.7749, -122.4194)


@pytest.mark.parametrize("bad", ["37.7749", "37.7749,-122.4194,0", "north,west", ""])
def test_parse_point_rejects_malformed_input(bad: str) -> None:
    # argparse's own error type, so a bad coordinate prints usage rather than a
    # traceback from inside the fetch.
    with pytest.raises(argparse.ArgumentTypeError, match="lat,lon"):
        climate.parse_point(bad)


def test_parse_points_file_skips_comments_and_blanks(tmp_path: Path) -> None:
    listing = tmp_path / "points.txt"
    listing.write_text(
        "# bay area sites\n"
        "37.7749,-122.4194\n"
        "\n"
        "37.78,-122.41  # trailing note\n"
        "# 37.9,-122.4  (commented out, not fetched)\n"
    )
    assert climate.parse_points_file(listing) == [
        (37.7749, -122.4194),
        (37.78, -122.41),
    ]


def test_parse_points_file_names_the_offending_line(tmp_path: Path) -> None:
    listing = tmp_path / "points.txt"
    listing.write_text("37.7749,-122.4194\n\nnorth,west\n")
    # The line number is the point of the message: a bare parse error in a file
    # of hundreds of coordinates says nothing about which one to fix.
    with pytest.raises(argparse.ArgumentTypeError, match="line 3"):
        climate.parse_points_file(listing)


def test_parse_points_file_rejects_a_file_with_no_points(tmp_path: Path) -> None:
    listing = tmp_path / "points.txt"
    listing.write_text("# every line commented out\n\n")
    with pytest.raises(argparse.ArgumentTypeError, match="no points found"):
        climate.parse_points_file(listing)


def test_points_from_args_combines_flags_and_files() -> None:
    args = argparse.Namespace(
        points=[(37.7749, -122.4194)],
        points_files=[[(37.78, -122.41), (37.9, -122.4)]],
    )
    # Duplicates and order are left alone -- consolidating is the fetch's job.
    assert climate.points_from_args(args) == [
        (37.7749, -122.4194),
        (37.78, -122.41),
        (37.9, -122.4),
    ]


def test_points_from_args_is_empty_when_neither_flag_is_given() -> None:
    args = argparse.Namespace(points=None, points_files=None)
    assert climate.points_from_args(args) == []
