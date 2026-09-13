"""Keep the ERA5-Land land-sea pre-check out of the way of every other test."""

from __future__ import annotations

import pytest

from external_data.climate_pipeline.era5 import land_mask, mask_testing


@pytest.fixture(autouse=True)
def _inert_land_mask(tmp_path_factory: pytest.TempPathFactory, monkeypatch):
    cache = tmp_path_factory.mktemp("era5_mask") / "mask.npy"
    monkeypatch.setenv(land_mask.CACHE_ENV, str(cache))
    mask_testing.install_all_land_mask(cache)
    yield
    land_mask.reset_cache()
