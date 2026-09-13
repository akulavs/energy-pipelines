"""
Tests for the recorded-failure contract.

Deliberately Prefect-free: the record is a persisted format, and the point of
moving it out of ``flows.py`` is that its rules can be exercised without spinning
a flow. Flow manifests are written directly.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from common.storage.flow_manifest import (
    FlowStatus,
    update_flow_manifest,
    write_flow_manifest_start,
)
from common.storage.manifest import ManifestRow
from external_data.climate_pipeline import dead_letter, point_manifest
from external_data.climate_pipeline.era5 import bronze as era5_bronze

_UTC = dt.timezone.utc
_DEAD = (0.0, -140.0)
_LIVE = (37.42, -122.23)


def _write_run(root: str, flow_name: str, failed: object, flow_id: str) -> None:
    """A completed flow manifest carrying `failed_points`, as a real run writes."""
    now = dt.datetime(2026, 1, 1, tzinfo=_UTC)
    write_flow_manifest_start(
        flow_id=flow_id,
        flow_name=flow_name,
        writer="test",
        root_uri=root,
        scheduled_time=now,
        start_time=now,
    )
    update_flow_manifest(
        flow_id=flow_id,
        root_uri=root,
        status=FlowStatus.COMPLETED,
        metadata={"failed_points": failed},
    )


def test_a_record_is_readable_and_parseable() -> None:
    entry = dead_letter.record(_DEAD, permanent=True, error="no data")
    assert entry["point"] == "(0.0, -140.0)"
    # Coordinates as their own fields, so matching never parses the display string.
    assert (float(entry["latitude"]), float(entry["longitude"])) == _DEAD
    assert entry["permanent"] == "True"


def test_known_points_reads_a_flat_bronze_record(tmp_path: Path) -> None:
    root = str(tmp_path)
    _write_run(
        root,
        "ingest_era5",
        [dead_letter.record(_DEAD, permanent=True, error="sea")],
        "a",
    )
    dead = dead_letter.known_points(root, "era5", era5_bronze.node)
    assert set(dead) == {era5_bronze.node(_DEAD)}


def test_known_points_reads_the_pipeline_shape(tmp_path: Path) -> None:
    # The pipeline records both sources under one key each; the bronze flows a
    # flat list. Both have to be readable or half the history is invisible.
    root = str(tmp_path)
    _write_run(
        root,
        "run_climate_pipeline",
        {"era5": [], "nsrdb": [dead_letter.record(_DEAD, permanent=True, error="4xx")]},
        "b",
    )
    assert set(dead_letter.known_points(root, "nsrdb", point_manifest.normalise)) == {
        point_manifest.normalise(_DEAD)
    }
    assert dead_letter.known_points(root, "era5", era5_bronze.node) == {}


def test_a_transient_record_is_not_dead(tmp_path: Path) -> None:
    # Only permanent failures suppress. A transient one may succeed next run.
    root = str(tmp_path)
    _write_run(
        root,
        "ingest_era5",
        [dead_letter.record(_DEAD, permanent=False, error="timeout")],
        "c",
    )
    assert dead_letter.known_points(root, "era5", era5_bronze.node) == {}


def test_a_record_without_coordinates_suppresses_nothing(tmp_path: Path) -> None:
    # Written before the coordinates were stored separately: unmatchable, and must
    # degrade to "no suppression" rather than raising.
    root = str(tmp_path)
    _write_run(
        root, "ingest_era5", [{"point": "(0.0, -140.0)", "permanent": "True"}], "d"
    )
    assert dead_letter.known_points(root, "era5", era5_bronze.node) == {}


def test_hold_back_splits_dead_from_live(tmp_path: Path) -> None:
    root = str(tmp_path)
    _write_run(
        root,
        "ingest_era5",
        [dead_letter.record(_DEAD, permanent=True, error="sea")],
        "e",
    )
    keep, held = dead_letter.hold_back(
        [_LIVE, _DEAD], root, "era5", era5_bronze.node, force_refresh=False
    )
    assert keep == [_LIVE]
    # Held, not dropped: the gap stays recorded for the silver decision.
    assert [e["point"] for e in held] == [f"({_DEAD[0]}, {_DEAD[1]})"]
    assert held[0]["permanent"] == "True"


def test_force_refresh_holds_nothing_back(tmp_path: Path) -> None:
    root = str(tmp_path)
    _write_run(
        root,
        "ingest_era5",
        [dead_letter.record(_DEAD, permanent=True, error="sea")],
        "f",
    )
    keep, held = dead_letter.hold_back(
        [_LIVE, _DEAD], root, "era5", era5_bronze.node, force_refresh=True
    )
    assert keep == [_LIVE, _DEAD]
    assert held == []


def test_unexplained_separates_recorded_gaps_from_never_ingested(
    tmp_path: Path,
) -> None:
    # The distinction silver builds on: a recorded gap is expected, an unrecorded
    # one usually means bronze was never run and should stop the build.
    root = str(tmp_path)
    _write_run(
        root,
        "ingest_era5",
        [dead_letter.record(_DEAD, permanent=True, error="sea")],
        "g",
    )
    missing = {era5_bronze.node(_DEAD), era5_bronze.node(_LIVE)}
    unexplained = dead_letter.unexplained(root, {"era5": (missing, era5_bronze.node)})
    assert unexplained == [f"era5 {era5_bronze.node(_LIVE)}"]


def test_no_records_explains_nothing(tmp_path: Path) -> None:
    missing = {era5_bronze.node(_LIVE)}
    assert dead_letter.unexplained(
        str(tmp_path), {"era5": (missing, era5_bronze.node)}
    ) == [f"era5 {era5_bronze.node(_LIVE)}"]


def _row(end: str) -> ManifestRow:
    """A manifest row keyed like a per-point NSRDB write ending on *end*."""
    return ManifestRow(
        dataset_name="nsrdb_solar_bronze",
        write_id="w",
        write_time=dt.datetime(2026, 1, 1, tzinfo=_UTC),
        writer="test",
        params_json=json.dumps(
            {
                "point": list(_LIVE),
                "start_date": "2023-01-01",
                "end_date": end,
                "interval": 60,
            }
        ),
        data_uri="mem://x",
    )


def test_a_write_reaching_the_requested_end_is_not_short() -> None:
    # The boundary: covering exactly what was asked for is a plain success. An
    # off-by-one here reports every complete fetch as a permanent gap.
    requested = dt.date(2024, 12, 31)
    assert dead_letter.short_writes([_row("2024-12-31")], requested) == []
    assert dead_letter.short_writes([_row("2025-06-30")], requested) == []


def test_a_narrowed_write_is_recorded_as_a_permanent_gap() -> None:
    # Permanent by construction: a transient year failure fails the whole point
    # rather than narrowing it, so a short write means the rest is unavailable.
    [entry] = dead_letter.short_writes([_row("2023-12-31")], dt.date(2024, 12, 31))
    assert (float(entry["latitude"]), float(entry["longitude"])) == _LIVE
    assert entry["permanent"] == "True"
    assert "covers only through 2023-12-31" in entry["error"]
