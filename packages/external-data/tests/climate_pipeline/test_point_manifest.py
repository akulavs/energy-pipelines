"""
Tests for the shared per-point manifest lookups.

Bronze is written one manifest entry per point, and these lookups are what make
that incremental: they decide whether a requested point already has a write
covering the range, or needs fetching. Exercised through real ERA5 writes rather
than hand-built manifest rows, so a change to the key shape fails here.

The ERA5 grain is the 0.1 deg **node**, not the exact coordinate -- CDS snaps to
it, so two nearby points share one download. Several tests below pin that, since
looking up at a finer grain than the write silently re-fetches.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from common.storage import columnar
from common.storage.manifest import ManifestRow
from external_data.climate_pipeline import point_manifest, schema
from external_data.climate_pipeline.era5 import bronze as era5_bronze

FIXTURES = Path(__file__).parent / "era5" / "fixtures"
BRONZE_PARQUET = FIXTURES / "era5_land_ts_bronze.parquet"

_START = dt.date(2025, 6, 1)
_END = dt.date(2025, 6, 1)
# Kept clear of the 0.05 deg cell boundaries: a coordinate sitting exactly on one
# (e.g. -122.25) rounds either way, so it would make these tests about rounding
# rather than about the lookup.
_A = (37.42, -122.23)
_B = (40.72, -73.96)
_C = (34.03, -118.22)
# A few metres from _A, so the same 0.1 deg node -> the same download.
_A_NEARBY = (37.421, -122.231)


@pytest.fixture(scope="session")
def golden() -> pd.DataFrame:
    return pd.read_parquet(BRONZE_PARQUET)


def _request(
    points: tuple[tuple[float, float], ...],
    start: dt.date = _START,
    end: dt.date = _END,
) -> schema.ClimatePipelineRequestArgs:
    return schema.ClimatePipelineRequestArgs(
        points=points, start_date=start, end_date=end
    )


def _write(
    golden: pd.DataFrame,
    root: str,
    point: tuple[float, float],
    start: dt.date = _START,
    end: dt.date = _END,
) -> None:
    """Write one point's bronze the way the flow does."""
    table = golden.assign(latitude=point[0], longitude=point[1])
    era5_bronze.write_point_bronze(
        table, point, _request((point,), start, end), root, writer="test"
    )


def _split(root: str, points, **kwargs):
    return point_manifest.split_by_coverage(
        points,
        era5_bronze.DATASET_NAME,
        root,
        point_manifest.read_point_key,
        kwargs.pop("start", _START),
        kwargs.pop("end", _END),
        grain=era5_bronze.node,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Grain
# --------------------------------------------------------------------------- #


def test_node_collapses_points_in_the_same_cell() -> None:
    assert era5_bronze.node(_A) == era5_bronze.node(_A_NEARBY)
    # Coarser than the coordinates it came from, and on the 0.1 deg grid.
    assert era5_bronze.node(_A) != point_manifest.normalise(_A)
    assert era5_bronze.node(_B) == (round(_B[0], 1), round(_B[1], 1))


def test_node_separates_points_in_different_cells() -> None:
    assert era5_bronze.node(_A) != era5_bronze.node(_B)


def test_normalise_rounds_to_the_comparison_precision() -> None:
    assert point_manifest.normalise((37.4200000001, -122.23)) == _A


# --------------------------------------------------------------------------- #
# Key readers
# --------------------------------------------------------------------------- #


def test_read_point_key_reads_a_single_point_key() -> None:
    params = {
        "points": [[37.42, -122.23]],
        "start_date": "2025-06-01",
        "end_date": "2025-06-02",
    }
    point, start, end = point_manifest.read_point_key(params)
    assert point == _A
    assert (start, end) == (dt.date(2025, 6, 1), dt.date(2025, 6, 2))


def test_read_point_key_rejects_a_batch_key() -> None:
    # A pre-per-point write holds several points in one entry. Reading it as its
    # first point would claim coverage for one location and lose the rest.
    params = {
        "points": [[37.42, -122.23], [40.72, -73.96]],
        "start_date": "2025-06-01",
        "end_date": "2025-06-01",
    }
    with pytest.raises(ValueError, match="not a per-point key"):
        point_manifest.read_point_key(params)


def test_read_point_key_unwraps_the_request() -> None:
    params = {
        "request": {
            "points": [[37.42, -122.23]],
            "start_date": "2025-06-01",
            "end_date": "2025-06-01",
        },
        "interval": 60,
    }
    assert point_manifest.read_point_key(params)[0] == _A


# --------------------------------------------------------------------------- #
# Coverage
# --------------------------------------------------------------------------- #


def _entry(start: dt.date, end: dt.date):
    """A manifest entry spanning [start, end]; only the dates matter to covers()."""
    row = ManifestRow(
        write_id="w",
        dataset_name=era5_bronze.DATASET_NAME,
        write_time=dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
        params_json="{}",
        data_uri="d",
        writer="test",
    )
    return (row, start, end)


@pytest.mark.parametrize(
    ("have", "want", "expected"),
    [
        ((dt.date(2025, 1, 1), dt.date(2025, 12, 31)), (_START, _END), True),
        ((_START, _END), (_START, _END), True),
        ((dt.date(2025, 6, 2), _END), (_START, _END), False),  # starts too late
        ((_START, dt.date(2025, 5, 31)), (_START, _END), False),  # ends too early
    ],
)
def test_covers_requires_the_whole_requested_span(have, want, expected) -> None:
    assert point_manifest.covers(_entry(*have), want[0], want[1]) is expected


def test_covers_rejects_a_missing_entry() -> None:
    assert point_manifest.covers(None, _START, _END) is False


# --------------------------------------------------------------------------- #
# Scanning
# --------------------------------------------------------------------------- #


def test_existing_points_finds_one_entry_per_point(
    golden: pd.DataFrame, tmp_path: Path
) -> None:
    root = str(tmp_path)
    for point in (_A, _B, _C):
        _write(golden, root, point)
    have = point_manifest.existing_points(
        era5_bronze.DATASET_NAME, root, point_manifest.read_point_key
    )
    assert set(have) == {point_manifest.normalise(p) for p in (_A, _B, _C)}


def test_existing_points_keys_at_the_requested_grain(
    golden: pd.DataFrame, tmp_path: Path
) -> None:
    root = str(tmp_path)
    _write(golden, root, _A)
    have = point_manifest.existing_points(
        era5_bronze.DATASET_NAME,
        root,
        point_manifest.read_point_key,
        grain=era5_bronze.node,
    )
    assert set(have) == {era5_bronze.node(_A)}


def test_existing_points_ignores_a_batch_era_entry(
    golden: pd.DataFrame, tmp_path: Path
) -> None:
    # A multi-point write cannot answer "do I have this point?", so it is skipped
    # rather than misread as its first point.
    root = str(tmp_path)
    # via pandas the timestamp comes back naive/ns, so restore the schema's dtype
    batch = pl.from_pandas(golden).with_columns(
        pl.col("valid_time").dt.cast_time_unit("us").dt.replace_time_zone("UTC")
    )
    columnar.write_dataset(
        batch,
        era5_bronze.Era5LandBronzeSchema,
        era5_bronze.DATASET_NAME,
        _request((_A, _B)),
        root,
        writer="test",
    )
    have = point_manifest.existing_points(
        era5_bronze.DATASET_NAME, root, point_manifest.read_point_key
    )
    assert have == {}


# --------------------------------------------------------------------------- #
# The skip decision
# --------------------------------------------------------------------------- #


def test_split_fetches_everything_when_nothing_is_written(tmp_path: Path) -> None:
    to_fetch, reusable = _split(str(tmp_path), (_A, _B))
    assert to_fetch == [_A, _B]
    assert reusable == []


def test_split_skips_a_covered_point(golden: pd.DataFrame, tmp_path: Path) -> None:
    root = str(tmp_path)
    _write(golden, root, _A)
    to_fetch, reusable = _split(root, (_A,))
    assert to_fetch == []
    assert len(reusable) == 1


def test_split_fetches_only_the_new_points(
    golden: pd.DataFrame, tmp_path: Path
) -> None:
    root = str(tmp_path)
    _write(golden, root, _A)
    to_fetch, reusable = _split(root, (_A, _B, _C))
    assert to_fetch == [_B, _C]
    assert len(reusable) == 1


def test_split_matches_at_node_grain(golden: pd.DataFrame, tmp_path: Path) -> None:
    # The write is keyed by node, so a coordinate a few metres away is already
    # covered -- looking up at exact precision would re-download the same cell.
    root = str(tmp_path)
    _write(golden, root, _A)
    to_fetch, reusable = _split(root, (_A_NEARBY,))
    assert to_fetch == []
    assert len(reusable) == 1


def test_split_refetches_when_the_range_widens(
    golden: pd.DataFrame, tmp_path: Path
) -> None:
    root = str(tmp_path)
    _write(golden, root, _A)
    to_fetch, reusable = _split(
        root, (_A,), start=dt.date(2025, 5, 1), end=dt.date(2025, 6, 30)
    )
    assert to_fetch == [point_manifest.normalise(_A)]
    assert reusable == []


def test_split_reuses_a_write_that_spans_more_than_asked(
    golden: pd.DataFrame, tmp_path: Path
) -> None:
    root = str(tmp_path)
    _write(golden, root, _A, start=dt.date(2025, 1, 1), end=dt.date(2025, 12, 31))
    to_fetch, reusable = _split(root, (_A,))
    assert to_fetch == []
    assert len(reusable) == 1


def test_split_force_refresh_ignores_what_exists(
    golden: pd.DataFrame, tmp_path: Path
) -> None:
    root = str(tmp_path)
    _write(golden, root, _A)
    to_fetch, reusable = _split(root, (_A,), force_refresh=True)
    assert to_fetch == [point_manifest.normalise(_A)]
    assert reusable == []


def test_split_force_refresh_does_not_scan(tmp_path: Path) -> None:
    # Nothing written, so a scan would be the only way to answer -- force_refresh
    # short-circuits before it, which is what makes it safe on an empty root.
    to_fetch, reusable = _split(str(tmp_path), (_A,), force_refresh=True)
    assert to_fetch == [point_manifest.normalise(_A)]
    assert reusable == []


# --------------------------------------------------------------------------- #
# Error rendering
# --------------------------------------------------------------------------- #


def test_describe_points_lists_them_in_order() -> None:
    assert (
        point_manifest.describe_points([_B, _A]) == "(37.42, -122.23), (40.72, -73.96)"
    )


def test_describe_points_truncates_a_long_list() -> None:
    points = [(float(i), float(i)) for i in range(9)]
    rendered = point_manifest.describe_points(points)
    assert rendered.count("(") == 5
    assert rendered.endswith("and 4 more")


# --------------------------------------------------------------------------- #
# One point, several ranges: which entry wins
# --------------------------------------------------------------------------- #

_WIDE_END = dt.date(2025, 6, 30)
_EARLIER = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
_LATER = dt.datetime(2026, 6, 1, tzinfo=dt.UTC)


def _write_at(
    golden: pd.DataFrame,
    root: str,
    point: tuple[float, float],
    end: dt.date,
    when: dt.datetime,
) -> None:
    """One point's bronze keyed to *end*, stamped with an explicit write time."""
    era5_bronze.write_point_bronze(
        golden.assign(latitude=point[0], longitude=point[1]),
        point,
        _request((point,), _START, end),
        root,
        writer="test",
        write_time=when,
    )


def test_a_later_wider_write_supersedes_an_earlier_narrow_one(
    golden: pd.DataFrame, tmp_path: Path
) -> None:
    # The sequence a partial fetch produces: a run that could only reach part of
    # the range, then a later one that got the rest. Resolving to the earlier,
    # narrower entry would hide bronze that exists -- the point would re-fetch on
    # every run, and silver would read the short file while the full one sat there.
    root = str(tmp_path)
    _write_at(golden, root, _A, _START, _EARLIER)
    _write_at(golden, root, _A, _WIDE_END, _LATER)

    have = point_manifest.existing_points(
        era5_bronze.DATASET_NAME,
        root,
        point_manifest.read_point_key,
        grain=era5_bronze.node,
    )
    entry = have[era5_bronze.node(_A)]
    assert entry[2] == _WIDE_END
    assert point_manifest.covers(entry, _START, _WIDE_END)

    # ...so the point is reused rather than fetched again.
    to_fetch, reusable = _split(root, [_A], end=_WIDE_END)
    assert to_fetch == []
    assert len(reusable) == 1


def test_the_most_recent_write_wins_even_when_it_is_narrower(
    golden: pd.DataFrame, tmp_path: Path
) -> None:
    # The documented contract, and the consistent extension of "the latest write
    # for these params wins": recency decides, not breadth. A narrower latest
    # entry therefore reads as partial coverage and the range is fetched again --
    # safe, because it never claims a span it does not hold.
    root = str(tmp_path)
    _write_at(golden, root, _A, _WIDE_END, _EARLIER)
    _write_at(golden, root, _A, _START, _LATER)

    have = point_manifest.existing_points(
        era5_bronze.DATASET_NAME,
        root,
        point_manifest.read_point_key,
        grain=era5_bronze.node,
    )
    assert have[era5_bronze.node(_A)][2] == _START

    to_fetch, reusable = _split(root, [_A], end=_WIDE_END)
    assert to_fetch == [point_manifest.normalise(_A)]
    assert reusable == []


def test_write_order_does_not_decide_which_entry_wins(
    golden: pd.DataFrame, tmp_path: Path
) -> None:
    # Pinning the mechanism, not just the outcome: the winner is chosen by write
    # time, so the order rows come back from the manifest scan cannot change it.
    # Depending on that order is what previously kept the oldest entry per point.
    wide_first = str(tmp_path / "wide-first")
    narrow_first = str(tmp_path / "narrow-first")
    # Same two writes, same timestamps, opposite insertion order.
    _write_at(golden, wide_first, _A, _WIDE_END, _LATER)
    _write_at(golden, wide_first, _A, _START, _EARLIER)
    _write_at(golden, narrow_first, _A, _START, _EARLIER)
    _write_at(golden, narrow_first, _A, _WIDE_END, _LATER)

    ends = [
        point_manifest.existing_points(
            era5_bronze.DATASET_NAME,
            root,
            point_manifest.read_point_key,
            grain=era5_bronze.node,
        )[era5_bronze.node(_A)][2]
        for root in (wide_first, narrow_first)
    ]
    assert ends == [_WIDE_END, _WIDE_END]


# --------------------------------------------------------------------------- #
# Key shape: the point itself, and the collection-of-one it replaced
# --------------------------------------------------------------------------- #


def test_reads_the_single_point_key() -> None:
    # The shape written now: the point, not a collection holding one.
    key = schema.PointSliceKey(
        point=(37.42, -122.23), start_date=_START, end_date=_WIDE_END
    )
    assert point_manifest.read_point_key(json.loads(key.model_dump_json())) == (
        (37.42, -122.23),
        _START,
        _WIDE_END,
    )


def test_reads_the_nsrdb_key_without_unwrapping_a_request() -> None:
    key = schema.NsrdbPointSliceKey(
        point=(37.77, -122.42),
        start_date=_START,
        end_date=_WIDE_END,
        interval=60,
        attributes=schema.NSRDB_ATTRIBUTES,
    )
    params = json.loads(key.model_dump_json())
    assert params["point"] == [37.77, -122.42]  # flat, not nested under "request"
    assert point_manifest.read_point_key(params)[0] == (37.77, -122.42)


def test_still_reads_the_legacy_collection_of_one() -> None:
    # Entries written before the key became single-point are still on disk.
    # Refusing them would orphan every earlier write and re-fetch the lot.
    legacy = {
        "points": [[37.42, -122.23]],
        "start_date": _START.isoformat(),
        "end_date": _WIDE_END.isoformat(),
    }
    assert point_manifest.read_point_key(legacy) == (
        (37.42, -122.23),
        _START,
        _WIDE_END,
    )


def test_still_reads_the_legacy_nested_nsrdb_key() -> None:
    legacy = {
        "request": {
            "points": [[37.77, -122.42]],
            "start_date": _START.isoformat(),
            "end_date": _WIDE_END.isoformat(),
        },
        "interval": 60,
    }
    assert point_manifest.read_point_key(legacy)[0] == (37.77, -122.42)


def test_still_rejects_a_legacy_batch_key() -> None:
    # A batch-era entry covers several locations in one write; claiming it for its
    # first point would report coverage for one and silently lose the rest.
    legacy = {
        "points": [[37.42, -122.23], [40.72, -73.96]],
        "start_date": _START.isoformat(),
        "end_date": _WIDE_END.isoformat(),
    }
    with pytest.raises(ValueError, match="not a per-point key"):
        point_manifest.read_point_key(legacy)
