"""
Tests for the load pipeline's ledger-coverage checks.

Coverage here is key *existence*, not a range comparison, so the thing worth
pinning is that it agrees with ``resolve_manifest`` -- a hit here must mean a hit
there, or the planner would skip a fetch whose data no reader can resolve.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from common.storage import columnar
from common.storage.manifest import resolve_manifest
from external_data.load_pipeline import coverage
from external_data.load_pipeline.resstock import bronze

_PUMAS = ("G11000101", "G11000105", "G11000103")


def _params(puma: str) -> bronze.ResstockPumaTimeseriesRequestArgs:
    return bronze.ResstockPumaTimeseriesRequestArgs(puma_gisjoin=puma)


def _write(puma: str, root: Path) -> None:
    frame = pl.DataFrame(
        {
            "timestamp": [dt.datetime(2018, 1, 1, 0, 15)],
            "bldg_id": [1],
            "state": ["DC"],
            "puma_gisjoin": [puma],
            "electricity_total_kwh": [1.0],
            "electricity_cooling_kwh": [0.0],
            "electricity_heating_kwh": [0.0],
        }
    )
    columnar.write_dataset(
        bronze.ResstockTimeseriesBronzeSchema.DataFrame(frame).cast(),
        bronze.ResstockTimeseriesBronzeSchema,
        bronze.TIMESERIES_DATASET_NAME,
        _params(puma),
        str(root),
        writer="test",
    )


@pytest.fixture
def seeded(tmp_path: Path) -> Path:
    """One PUMA written, two not."""
    _write(_PUMAS[0], tmp_path)
    return tmp_path


def test_an_empty_root_covers_nothing(tmp_path: Path) -> None:
    to_write, covered = coverage.split_by_coverage(
        bronze.TIMESERIES_DATASET_NAME, [_params(p) for p in _PUMAS], str(tmp_path)
    )
    assert len(to_write) == 3
    assert covered == []


def test_split_separates_written_from_missing(seeded: Path) -> None:
    to_write, covered = coverage.split_by_coverage(
        bronze.TIMESERIES_DATASET_NAME, [_params(p) for p in _PUMAS], str(seeded)
    )
    assert [p.puma_gisjoin for p in to_write] == list(_PUMAS[1:])
    assert len(covered) == 1


def test_a_covered_key_resolves_for_a_reader(seeded: Path) -> None:
    """
    The invariant the planner depends on: skipping a fetch is only safe if a
    reader can resolve what was skipped. Both use ``params_json`` equality, so
    this holds by construction -- and would break loudly if either side changed
    how it serialises a key.
    """
    row = coverage.covered_write(
        bronze.TIMESERIES_DATASET_NAME, _params(_PUMAS[0]), str(seeded)
    )
    assert row is not None
    resolved = resolve_manifest(
        dataset_name=bronze.TIMESERIES_DATASET_NAME,
        params=_params(_PUMAS[0]),
        root_uri=str(seeded),
    )
    assert resolved.write_id == row.write_id


def test_a_missing_key_is_none(seeded: Path) -> None:
    assert (
        coverage.covered_write(
            bronze.TIMESERIES_DATASET_NAME, _params(_PUMAS[1]), str(seeded)
        )
        is None
    )


def test_force_refresh_reports_everything_as_missing(seeded: Path) -> None:
    to_write, covered = coverage.split_by_coverage(
        bronze.TIMESERIES_DATASET_NAME,
        [_params(p) for p in _PUMAS],
        str(seeded),
        force_refresh=True,
    )
    assert len(to_write) == 3
    assert covered == []


def test_a_differing_field_is_a_different_key(seeded: Path) -> None:
    """
    Coverage is exact, not fuzzy: an upgrade the ledger has never seen must not
    read as covered by the baseline's write.
    """
    other = bronze.ResstockPumaTimeseriesRequestArgs(puma_gisjoin=_PUMAS[0], upgrade=2)
    assert (
        coverage.covered_write(bronze.TIMESERIES_DATASET_NAME, other, str(seeded))
        is None
    )
