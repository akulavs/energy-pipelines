"""
Tests for the ERA5-Land land-sea pre-check.

Never touches CDS: each test points the cache at a synthetic mask via the
``ERA5_LAND_MASK_CACHE`` env var, so the grid arithmetic and the fallback are
exercised without a download.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from external_data.climate_pipeline import failures, schema
from external_data.climate_pipeline.era5 import bronze, land_mask


BRONZE_PARQUET = Path(__file__).parent / "fixtures" / "era5_land_ts_bronze.parquet"


@pytest.fixture(scope="session")
def bronze_golden() -> pd.DataFrame:
    return pd.read_parquet(BRONZE_PARQUET)


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(land_mask.CACHE_ENV, str(tmp_path / "mask.npy"))
    land_mask.reset_cache()
    yield
    land_mask.reset_cache()


def _write_mask(land_cells: set[tuple[int, int]]) -> None:
    """Cache a mask that is sea everywhere except the given (row, column) cells."""
    grid = np.zeros((land_mask.LAT_COUNT, land_mask.LON_COUNT), dtype=bool)
    for row, column in land_cells:
        grid[row, column] = True
    land_mask._store(grid)
    land_mask.reset_cache()


def _cell(latitude: float, longitude: float) -> tuple[int, int]:
    row = int(round((land_mask.LAT_ORIGIN - latitude) / land_mask.GRID_STEP))
    column = int(round((longitude % 360.0) / land_mask.GRID_STEP)) % land_mask.LON_COUNT
    return row, column


def test_a_cell_with_no_land_is_sea() -> None:
    _write_mask({_cell(37.4, -122.2)})
    assert land_mask.is_sea((25.0, -45.0)) is True


def test_a_cell_with_land_is_not_sea() -> None:
    _write_mask({_cell(37.4, -122.2)})
    assert land_mask.is_sea((37.4, -122.2)) is False


def test_negative_longitudes_map_onto_the_published_grid() -> None:
    # The mask is published 0..360 while requests use -180..180; getting this
    # wrong would mirror the globe and reject land while accepting ocean.
    _write_mask({_cell(51.5, -0.12), _cell(-33.9, 151.2)})
    assert land_mask.is_sea((51.5, -0.12)) is False  # London, west of Greenwich
    assert land_mask.is_sea((-33.9, 151.2)) is False  # Sydney, east
    assert land_mask.is_sea((51.5, 179.9)) is True  # antipodal-ish, still sea


def test_an_unobtainable_mask_never_blocks_a_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The mask is an optimisation, not a gate: without credentials the pre-check
    # has to abstain rather than refuse every point on Earth.
    def _boom(*args, **kwargs):
        raise RuntimeError("no CDS credentials")

    monkeypatch.setattr(land_mask, "_download_mask", _boom)
    assert land_mask.is_sea((25.0, -45.0)) is False
    assert land_mask.load() is None


def test_the_mask_is_read_from_cache_once(monkeypatch: pytest.MonkeyPatch) -> None:
    # Every point in a fan-out asks; re-reading the file per point would trade the
    # saved request for a slower kind of waste.
    _write_mask({_cell(37.4, -122.2)})
    reads: list[int] = []
    original = land_mask._load_cached

    def _counted():
        reads.append(1)
        return original()

    monkeypatch.setattr(land_mask, "_load_cached", _counted)
    land_mask.reset_cache()
    for _ in range(25):
        land_mask.is_sea((37.4, -122.2))
    assert len(reads) == 1


def test_fetch_refuses_a_sea_point_without_contacting_cds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The whole point: no CDS request is made at all, so the failure costs
    # milliseconds instead of a queue slot and a minute.
    _write_mask({_cell(37.4, -122.2)})
    called: list[object] = []
    monkeypatch.setattr(bronze, "download_to_scratch", lambda *a, **k: called.append(1))

    request = schema.ClimatePipelineRequestArgs(points=((25.0, -45.0),))
    with pytest.raises(failures.PermanentFetchError, match="no land at"):
        bronze.fetch_point_table((25.0, -45.0), request)
    assert not called


def test_fetch_still_calls_cds_for_a_land_point(
    monkeypatch: pytest.MonkeyPatch, bronze_golden: pd.DataFrame
) -> None:
    _write_mask({_cell(37.4, -122.2)})
    monkeypatch.setattr(
        bronze, "_fetch_point_table", lambda *a, **k: bronze_golden.copy()
    )
    request = schema.ClimatePipelineRequestArgs(points=((37.4, -122.2),))
    assert not bronze.fetch_point_table((37.4, -122.2), request).empty


def test_a_truncated_cache_refetches_rather_than_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A short-but-loadable file survives np.load and only fails on the reshape.
    # That has to warn and refetch: raising here would escape `is_sea` and fail the
    # point, contradicting the module's "never a gate" promise.
    _write_mask({_cell(37.4, -122.2)})
    path = land_mask.cache_path()
    np.save(path, np.load(path)[:100])  # truncate
    land_mask.reset_cache()

    rebuilt = np.zeros((land_mask.LAT_COUNT, land_mask.LON_COUNT), dtype=bool)
    monkeypatch.setattr(land_mask, "_download_mask", lambda *a, **k: rebuilt)

    assert land_mask.is_sea((37.4, -122.2)) is True  # refetched, not raised
    assert land_mask.load() is not None


def test_a_cold_cache_downloads_once_across_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # is_sea runs per point on up to 32 fan-out workers. Without the lock the
    # first arrivals each pull the whole ~50 MB mask, spending the CDS slots the
    # pre-check exists to save.
    import threading

    downloads: list[int] = []

    def _slow_download(*args, **kwargs):
        downloads.append(1)
        time.sleep(0.2)  # widen the window every unlocked thread would race through
        return np.ones((land_mask.LAT_COUNT, land_mask.LON_COUNT), dtype=bool)

    monkeypatch.setattr(land_mask, "_download_mask", _slow_download)
    land_mask.reset_cache()

    barrier = threading.Barrier(8)

    def _worker():
        barrier.wait()
        land_mask.is_sea((37.4, -122.2))

    threads = [threading.Thread(target=_worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert downloads == [1]


def test_the_cache_file_is_replaced_atomically(tmp_path: Path) -> None:
    # A half-written cache is indistinguishable from a corrupt one, so the write
    # lands on a sibling and is renamed into place.
    seen: list[str] = []
    real_replace = os.replace

    def _watch(src, dst):
        seen.append(str(dst))
        return real_replace(src, dst)

    with mock.patch.object(os, "replace", _watch):
        land_mask._store(
            np.ones((land_mask.LAT_COUNT, land_mask.LON_COUNT), dtype=bool)
        )

    assert seen == [str(land_mask.cache_path())]
    # No temp files left behind.
    assert not list(land_mask.cache_path().parent.glob("*.npy.tmp*"))
