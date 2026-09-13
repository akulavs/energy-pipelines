"""
ERA5-Land's own land-sea mask, to refuse a sea point before paying to fetch it.

CDS does not error on a point at sea: it queues, runs, and returns the full time
axis with every variable missing. That is ~40 s and a CDS slot to learn nothing,
and the emptiness only surfaces on the way out (``bronze._require_values``).

The mask is the model's own ``lsm`` field on the same 0.1 deg grid the data is
served on, so there is no coastline approximation to disagree with. Downloaded
once (~50 MB), reduced to a packed "has any land" bitmask (~790 KB) and cached.

**Only certain sea is refused** -- ``lsm`` exactly zero, 64% of the globe. Cells
with any land fraction, including the ~2% below 0.5, are still fetched and
``_require_values`` catches the empty ones. Refusing those here would risk
discarding real data to save a request.

Never a gate: if the mask cannot be obtained (no credentials, offline, failed
download) every lookup abstains and the fetch proceeds as before.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
from pathlib import Path
from collections.abc import Sequence

import cdsapi
import numpy as np

logger = logging.getLogger(__name__)

# The mask is static, so one timestamp is enough.
MASK_DATASET = "reanalysis-era5-land"
MASK_VARIABLE = "land_sea_mask"
_MASK_REQUEST = {
    "variable": MASK_VARIABLE,
    "year": "2020",
    "month": "01",
    "day": "01",
    "time": "00:00",
    "data_format": "netcdf",
}

# The grid the mask is published on, and that ERA5-Land serves data on. Latitude
# runs north to south; longitude is 0..360, not -180..180.
GRID_STEP = 0.1
LAT_ORIGIN = 90.0
LAT_COUNT = 1801
LON_COUNT = 3600

CACHE_ENV = "ERA5_LAND_MASK_CACHE"

# Sentinel distinguishing "not looked up yet" from "looked up, unavailable".
_UNLOADED = object()
_cache: object | np.ndarray | None = _UNLOADED
# Every point in a fan-out calls in, on up to 32 worker threads. Without this the
# first cold lookups would each download the same ~50 MB, spending the very CDS
# slots the pre-check exists to save.
_lock = threading.Lock()


def cache_path() -> Path:
    """Where the reduced mask is kept between runs."""
    override = os.environ.get(CACHE_ENV)
    if override:
        return Path(override)
    root = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(root) / "energy-pipelines" / "climate_pipeline" / "era5_land_lsm.npy"


def _download_mask(client: cdsapi.Client | None = None) -> np.ndarray:
    """Fetch ``lsm`` from CDS and reduce it to a "has any land" boolean grid."""
    import xarray as xr

    from external_data.climate_pipeline.era5 import bronze

    resolved = client if client is not None else cdsapi.Client()
    scratch = Path(tempfile.mkdtemp(prefix="era5_land_mask_"))
    try:
        target = scratch / "lsm.download"
        resolved.retrieve(MASK_DATASET, _MASK_REQUEST, str(target))
        # A zip around one NetCDF; the bronze helper resolves either shape.
        (netcdf,) = bronze.extract_netcdfs(target, scratch)
        with xr.open_dataset(netcdf) as dataset:
            lsm = np.asarray(dataset["lsm"].squeeze().values)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    if lsm.shape != (LAT_COUNT, LON_COUNT):
        msg = f"unexpected ERA5-Land mask shape {lsm.shape}, wanted {(LAT_COUNT, LON_COUNT)}"
        raise ValueError(msg)
    # Any land at all, not a majority: only certain sea is refused.
    return lsm > 0


def _load_cached() -> np.ndarray | None:
    path = cache_path()
    if not path.exists():
        return None
    try:
        packed = np.load(path)
        # Unpacking is inside the guard too: a truncated file loads fine and only
        # fails on the reshape, and that must refetch rather than escape as a
        # ValueError from ``is_sea``.
        return np.unpackbits(packed).reshape(LAT_COUNT, LON_COUNT).astype(bool)
    except OSError, ValueError:
        logger.warning("ERA5-Land mask cache at %s is unreadable; refetching", path)
        return None


def _store(mask: np.ndarray) -> None:
    path = cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written to a sibling then renamed: a crash or a concurrent writer would
    # otherwise leave a half-written file that reads as a corrupt cache. Packed,
    # so the artefact is ~790 KB rather than ~50 MB.
    handle, tmp = tempfile.mkstemp(dir=path.parent, suffix=".npy")
    os.close(handle)
    try:
        np.save(tmp, np.packbits(mask))
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def load(client: cdsapi.Client | None = None) -> np.ndarray | None:
    """
    The "has any land" grid, from cache or CDS, or ``None`` if unobtainable.

    Held in module state, behind a lock: every point in a fan-out asks, and the
    threads that arrive first would otherwise each download the whole mask.
    """
    global _cache
    if _cache is not _UNLOADED:
        return _cache  # ty:ignore[invalid-return-type]

    with _lock:
        # Re-checked: another thread may have finished while this one waited.
        if _cache is not _UNLOADED:
            return _cache  # ty:ignore[invalid-return-type]

        mask = _load_cached()
        if mask is None:
            try:
                mask = _download_mask(client)
            except Exception as exc:  # noqa: BLE001 - the mask is never a gate
                logger.warning(
                    "could not obtain the ERA5-Land land-sea mask (%s); points "
                    "will be fetched without the pre-check",
                    exc,
                )
                _cache = None
                return None
            _store(mask)
            logger.info("cached the ERA5-Land land-sea mask at %s", cache_path())

        _cache = mask
        return mask


def reset_cache() -> None:
    """Forget the in-memory mask. For tests, and after replacing the cached file."""
    global _cache
    _cache = _UNLOADED


def is_sea(point: Sequence[float], client: cdsapi.Client | None = None) -> bool:
    """
    Is this point in a cell ERA5-Land holds **no** land for?

    ``True`` only when certain. An unobtainable mask or any land fraction both
    give ``False`` -- fetch, and let the response decide.
    """
    mask = load(client)
    if mask is None:
        return False
    latitude, longitude = float(point[0]), float(point[1])
    # Latitude descends from +90; longitude is stored 0..360.
    row = int(round((LAT_ORIGIN - latitude) / GRID_STEP))
    column = int(round((longitude % 360.0) / GRID_STEP)) % LON_COUNT
    if not (0 <= row < LAT_COUNT):
        return False
    return not bool(mask[row, column])
