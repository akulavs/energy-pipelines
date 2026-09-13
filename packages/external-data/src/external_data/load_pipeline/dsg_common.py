"""
Shared machinery for the dsgrid-legacy EFS ``.dsg`` HDF5 files (2018 EFS)
"""

from __future__ import annotations
import datetime as dt
import io
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple
import h5py
import numpy as np
import polars as pl
import httpx
import common.exceptions
from common.frames import BaseDataFrameSchema
from external_data.load_pipeline.retry import retry_wait

logger = logging.getLogger(__name__)

# Public OEDI data lake (no account, API key, or login required)
BASE_URL = "https://oedi-data-lake.s3.amazonaws.com"
OEDI_PREFIX = "dsgrid-2018-efs/raw_complete"
SOURCE_DATASET = "dsgrid-2018-efs"

# 2012 hourly weather year (leap: 8784 hours), interval-ending, fixed -05:00
# (EST, no DST) -- stored naive local standard.
N_HOURS = 8784

# uint32 max is the dsgrid "null" sentinel: the sector is absent in that county
NULL_IDX = np.iinfo(np.uint32).max

STATE_PATTERN = r"^[A-Z]{2}$"


# --------------------------------------------------------------------------- #
# Fetch (raw)
# --------------------------------------------------------------------------- #


def download_dsg(
    url: str,
    *,
    max_retries: int = 3,
    backoff_seconds: float = 2.0,
    client: httpx.Client | None = None,
) -> bytes:
    """
    Download the ``.dsg`` HDF5 file from the public OEDI data lake, returning bytes

    No credentials are required; the bucket is fully public.
    """
    owns_client = client is None
    client = (
        client
        if client is not None
        else httpx.Client(timeout=300, follow_redirects=True)
    )
    last_exc: Exception | None = None
    try:
        for attempt in range(1, max_retries + 1):
            logger.info(
                "requesting dsgrid file (%s, attempt=%d/%d)", url, attempt, max_retries
            )
            try:
                response = client.get(url)
            except httpx.RequestError as exc:
                last_exc = exc
                if attempt < max_retries:
                    wait = backoff_seconds * (2 ** (attempt - 1))
                    logger.warning("request failed (%s); retrying in %.1fs", exc, wait)
                    time.sleep(wait)
                continue

            # 4xx are caller errors (bad path / release) -- except 429, below
            if 400 <= response.status_code < 500 and response.status_code != 429:
                raise common.exceptions.PipelineValueError(
                    f"OEDI rejected the request ({response.status_code}) for {url}: "
                    f"{response.text[:200].strip()}"
                )
            # 429 is transient like 5xx. Mirrors the sibling helper in
            # oedi_building_stock; these .dsg files are the largest objects the
            # pipeline pulls, which is when a server most likely asks for a pause.
            if response.status_code == 429 or response.status_code >= 500:
                last_exc = httpx.HTTPStatusError(
                    f"transient status {response.status_code}",
                    request=response.request,
                    response=response,
                )
                if attempt < max_retries:
                    wait = retry_wait(response, backoff_seconds, attempt)
                    logger.warning(
                        "transient status %d; retrying in %.1fs",
                        response.status_code,
                        wait,
                    )
                    time.sleep(wait)
                continue

            return response.content
    finally:
        if owns_client:
            client.close()

    raise common.exceptions.PipelineError(
        f"failed to download {url} after {max_retries} attempts"
    ) from last_exc


def load_dsg_bytes(
    url: str, dsg_bytes: bytes | None, client: httpx.Client | None
) -> bytes:
    """Return ``dsg_bytes`` if provided, else download from ``url``."""
    return dsg_bytes if dsg_bytes is not None else download_dsg(url, client=client)


# --------------------------------------------------------------------------- #
# Reconstruct -> per-(county x sector) hourly MWh arrays
# --------------------------------------------------------------------------- #


def fips_to_county_gisjoin(fips: str) -> str:
    """
    Convert a 5-digit county FIPS to its NHGIS county GISJOIN

    ``"11001"`` (DC) -> ``"G1100010"`` -- ``G`` + 2-digit state + ``0`` + 3-digit
    county + ``0``
    """
    return f"G{fips[:2]}0{fips[2:]}0"


def _decode(values: np.ndarray) -> list[str]:
    return [v.decode("utf-8") for v in values]


def parse_timestamps(raw: list[str]) -> list[dt.datetime]:
    """Parse the ISO timestamps to naive local-standard datetimes."""
    return [dt.datetime.fromisoformat(t).replace(tzinfo=None) for t in raw]


class Enumerations:
    """Decoded enumeration tables shared by every sector group in the file"""

    def __init__(self, root: h5py.Group) -> None:
        if "enumerations" not in root or "data" not in root:
            raise common.exceptions.PipelineValueError(
                "dsgrid file is missing the 'enumerations'/'data' groups; "
                "not a recognized .dsg file"
            )
        enums = root["enumerations"]
        self.geo_fips = _decode(enums["geography"]["id"][...])
        self.geo_name = _decode(enums["geography"]["name"][...])
        self.enduse_ids = _decode(enums["enduse"]["id"][...])
        sector = enums["sector"][...]
        self.sector_name = {
            sid.decode(): nm.decode()
            for sid, nm in zip(sector["id"], sector["name"], strict=True)
        }
        self.timestamps = parse_timestamps(_decode(enums["time"]["id"][...]))
        # county -> two-letter state, parsed from the "..., ST" name suffix and
        # validated rather than assumed: this is the pipeline's whole state selector,
        # and a malformed name would otherwise surface later as "no counties found
        # for state XX", blaming the caller for a naming problem in the file.
        self.geo_state = []
        for name in self.geo_name:
            if len(name) < 4 or name[-4:-2] != ", " or not name[-2:].isupper():
                msg = (
                    f"county name {name!r} does not end in the expected ', ST' "
                    "suffix, so its state cannot be determined"
                )
                raise common.exceptions.PipelineValueError(msg)
            self.geo_state.append(name[-2:])


class Segment(NamedTuple):
    """One reconstructed county x sector unit"""

    county_idx: int
    naics: str  # the data-group key: 4-digit subsector or 2-digit sector
    hourly: np.ndarray  # (n_enduses, n_hours) MWh


def county_indices_for_state(enums: Enumerations, state: str) -> list[int]:
    idx = [i for i, st in enumerate(enums.geo_state) if st == state]
    if not idx:
        raise common.exceptions.PipelineValueError(
            f"no counties found for state {state} in the dsgrid file"
        )
    return idx


def reconstruct_state(
    data: bytes, state: str, expected_enduse_ids: tuple[str, ...]
) -> tuple[Enumerations, list[Segment]]:
    """
    Reconstruct every (county x sector) segment for *state* into per-enduse
    hourly MWh arrays

    ``expected_enduse_ids`` is the end-use enumeration the caller's schema
    assumes (the file is validated against it, fail-loud). Sectors absent from
    all of the state's counties are skipped.
    """
    n_enduses = len(expected_enduse_ids)
    with h5py.File(io.BytesIO(data), "r") as root:
        enums = Enumerations(root)
        if enums.enduse_ids != list(expected_enduse_ids):
            raise common.exceptions.PipelineValueError(
                f"unexpected end uses {enums.enduse_ids}; "
                f"expected {list(expected_enduse_ids)}"
            )
        county_idx = county_indices_for_state(enums, state)
        n_hours = len(enums.timestamps)
        # The block check below is against the file's own enumeration, so a source
        # re-published with a different axis in *both* places would validate and
        # silently change the year under silver. Warn rather than raise: the
        # committed fixtures carry a 168-hour slice on purpose, and the real axis is
        # asserted in the live integration test.
        if n_hours != N_HOURS:
            logger.warning(
                "time enumeration has %d hour(s), expected %d (2012 is a leap "
                "year); expected for a fixture slice, not for a full .dsg file",
                n_hours,
                N_HOURS,
            )

        segments: list[Segment] = []
        for naics in root["data"]:
            grp = root["data"][naics]
            gidx = grp["geographies"]["idx"][...]
            gscale = grp["geographies"]["scale"][...]
            enduse_scale = grp["enduses"]["scale"][...].reshape(-1, 1)
            time_scale = grp["times"]["scale"][...].reshape(1, -1)
            shapes: np.ndarray | None = None
            for ci in county_idx:
                shape_row = gidx[ci]
                if shape_row == NULL_IDX:
                    continue
                if shapes is None:
                    # Bind the h5py dataset, don't read it: ``shape`` is metadata, so
                    # the check costs nothing and ``shapes[shape_row]`` then reads one
                    # (n_enduses, n_hours) slice rather than the sector's whole block.
                    shapes = grp["data"]  # (n_shapes, n_enduses, n_hours)
                    if shapes.shape[1:] != (n_enduses, n_hours):
                        raise common.exceptions.PipelineValueError(
                            f"sector {naics} data block has shape {shapes.shape}; "
                            f"expected (*, {n_enduses}, {n_hours})"
                        )
                hourly = (
                    shapes[shape_row].astype(np.float64)
                    * gscale[ci]
                    * enduse_scale
                    * time_scale
                )
                segments.append(Segment(county_idx=ci, naics=naics, hourly=hourly))
    return enums, segments


# --------------------------------------------------------------------------- #
# Shape helpers
# --------------------------------------------------------------------------- #


def empty_frame(columns: Sequence[str]) -> pl.DataFrame:
    return pl.DataFrame({c: [] for c in columns})


def shape_to_schema(
    frame: pl.DataFrame,
    schema: type[BaseDataFrameSchema],
    columns: Sequence[str],
    key_columns: Sequence[str],
) -> pl.DataFrame:
    """
    Cast a frame to *schema*'s declared dtypes, order/sort, and validate -- so
    the schema is the single source of dtype truth
    """
    shaped = schema.DataFrame(frame.select(columns)).cast().sort(key_columns)
    schema.validate(shaped)
    return shaped


def resolve_root_uri(root_uri: str | Path) -> str:
    if "://" not in str(root_uri):
        return str(Path(root_uri).resolve())
    return str(root_uri)
