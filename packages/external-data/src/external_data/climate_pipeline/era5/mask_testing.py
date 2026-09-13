"""
Test helper for the ERA5-Land land-sea pre-check.

Lives beside the module it isolates rather than being duplicated in each test
tree's ``conftest.py``. The mask is cached in a user-level directory, so without
isolation a developer who has run the pipeline would have a real mask on disk and
tests would quietly start refusing the mid-ocean coordinates several of them use
as stand-ins for a permanently unserviceable point -- the pre-check doing its job,
but the tests measuring something other than what they claim.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from external_data.climate_pipeline.era5 import land_mask


def install_all_land_mask(cache: Path) -> None:
    """Point the cache at *cache* and seed it all-land, making the check inert."""
    land_mask.reset_cache()
    land_mask._store(np.ones((land_mask.LAT_COUNT, land_mask.LON_COUNT), dtype=bool))
    land_mask.reset_cache()
