"""
ResStock (residential building stock) -- bronze ingestion

Two bronze datasets, both pulled from the public OEDI data lake for the latest
ResStock release:

- ``resstock_metadata_bronze`` -- one row per building model of the
  ``metadata_and_annual_results`` file: building characteristics + annual
  end-use energy totals.
- ``resstock_timeseries_bronze`` -- the 15-minute electricity load profile
  for *every* building in a PUMA, assembled into one dataset (``bldg_id`` is a
  join-key column). Building ids are resolved from the metadata file and their
  individual ``timeseries_individual_buildings`` files fetched concurrently.
"""

from __future__ import annotations
import datetime as dt
import io
import logging
from collections.abc import Sequence
from pathlib import Path
import httpx
import patito as pt
import polars as pl
import pydantic
import common.exceptions
from common.frames import BaseDataFrameSchema
from common.storage import columnar
from common.storage.manifest import ManifestRow
from external_data.load_pipeline import failures
from external_data.load_pipeline import oedi_building_stock as oedi
from external_data.load_pipeline.oedi_building_stock import download_object

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Shared OEDI location constants re-exported for the module's public/test surface
BASE_URL = oedi.BASE_URL
OEDI_PREFIX = oedi.OEDI_PREFIX

# Latest ResStock release (Oct/Nov 2025)
RELEASE = "resstock_amy2018_release_1"
DATASET = RELEASE

DEFAULT_WRITER = "resstock_bronze"
DEFAULT_BASE_DIR = oedi.DEFAULT_BASE_DIR

# Raw metadata column carrying the PUMA GISJOIN
_METADATA_PUMA_RAW = "in.puma"


# --------------------------------------------------------------------------- #
# Request
# --------------------------------------------------------------------------- #


class ResstockRequestBase(oedi.OediBuildingStockRequestBase):
    """
    Shared release + geography fields for ResStock OEDI requests
    """

    source_dataset: str = pydantic.Field(
        default=DATASET,
        description="OEDI ResStock release directory the data was retrieved from",
    )


class ResstockMetadataRequestArgs(ResstockRequestBase):
    """
    Arguments describing a per-building metadata / annual-results request for one
    PUMA

    Source and write disagree deliberately: OEDI publishes this file per state, so
    :meth:`object_url` is state-derived and one download serves every PUMA in it, but
    the write holds only the requested PUMA's buildings and its key says so. That is
    what makes every dataset here addressable by the thing a caller has -- a PUMA --
    as the climate pipeline keys every write by its point.

    The cost falls on the fetch: a PUMA ingested in a later run re-downloads the state
    file to cut its own slice. One download still covers every PUMA of one run.
    """

    puma_gisjoin: str = pydantic.Field(
        default=oedi.DEFAULT_PUMA,
        pattern=oedi.PUMA_PATTERN,
        description="NHGIS PUMA GISJOIN whose buildings this write holds",
    )

    def object_url(self) -> str:
        """
        Full HTTPS URL of the state metadata parquet on the OEDI data lake

        State-derived: OEDI has no per-PUMA ResStock metadata file, so the PUMA in the
        key describes what was *kept*, not what was fetched.
        """
        return (
            f"{BASE_URL}/{OEDI_PREFIX}/{self.release_year}/{self.source_dataset}/"
            f"metadata_and_annual_results/by_state/full/parquet/"
            f"state={self.state}/{self.state}_upgrade{self.upgrade}.parquet"
        )


# --------------------------------------------------------------------------- #
# Dataset schemas (catalog / manifest)
# --------------------------------------------------------------------------- #

METADATA_DATASET_NAME = "resstock_metadata_bronze"

# Raw OEDI column -> curated bronze field
METADATA_RAW_TO_COLUMN: dict[str, str] = {
    "bldg_id": "bldg_id",
    "in.county": "county_gisjoin",
    "in.county_name": "county_name",
    "in.puma": "puma_gisjoin",
    "in.geometry_building_type_recs": "building_type",
    "in.vintage": "vintage",
    "in.hvac_cooling_type": "hvac_cooling_type",
    "in.heating_fuel": "heating_fuel",
    "in.ashrae_iecc_climate_zone_2004": "climate_zone",
    "in.sqft..ft2": "sqft",
    "weight": "weight",
    "out.electricity.total.energy_consumption..kwh": "annual_electricity_total_kwh",
    "out.electricity.cooling.energy_consumption..kwh": "annual_electricity_cooling_kwh",
    "out.electricity.heating.energy_consumption..kwh": "annual_electricity_heating_kwh",
}
METADATA_COLUMNS = list(METADATA_RAW_TO_COLUMN.values())
# ResStock is one row per building.
METADATA_KEY_COLUMNS = ["bldg_id"]
# No partition columns: a PUMA's buildings can span several counties, so
# partitioning a one-PUMA write would scatter it for nothing. The manifest is the
# index.

# Curated timeseries columns (ResStock keeps the ``..kwh`` suffix on the raw side).
TIMESERIES_RAW_TO_COLUMN: dict[str, str] = {
    "timestamp": "timestamp",
    "bldg_id": "bldg_id",
    "out.electricity.total.energy_consumption..kwh": "electricity_total_kwh",
    "out.electricity.cooling.energy_consumption..kwh": "electricity_cooling_kwh",
    "out.electricity.heating.energy_consumption..kwh": "electricity_heating_kwh",
}

# The timeseries bronze schema + column lists are shared
TIMESERIES_DATASET_NAME = "resstock_timeseries_bronze"
ResstockTimeseriesBronzeSchema = oedi.OediBuildingStockTimeseriesSchema
TIMESERIES_COLUMNS = oedi.TIMESERIES_COLUMNS
TIMESERIES_KEY_COLUMNS = oedi.TIMESERIES_KEY_COLUMNS


class ResstockMetadataBronzeSchema(BaseDataFrameSchema):
    """
    One row per building model of the ResStock metadata file
    """

    bldg_id: int = pt.Field(
        dtype=pl.Int64,
        description="ResStock building model id; join key, NOT stable across releases",
    )
    county_gisjoin: str = pt.Field(
        dtype=pl.String,
        description="NHGIS county GISJOIN (utility-service-territory proxy)",
    )
    county_name: str = pt.Field(dtype=pl.String, description="county name")
    puma_gisjoin: str = pt.Field(
        dtype=pl.String, description="NHGIS PUMA GISJOIN (finer sub-state geography)"
    )
    building_type: str = pt.Field(
        dtype=pl.String,
        description="ResStock residential building type (spaced Title Case enum)",
    )
    vintage: str = pt.Field(dtype=pl.String, description="building age cohort")
    hvac_cooling_type: str = pt.Field(
        dtype=pl.String,
        description="residential cooling equipment type ('None' if no cooling)",
    )
    heating_fuel: str = pt.Field(
        dtype=pl.String,
        description="primary heating fuel (spaced Title Case; literal 'None' if none)",
    )
    climate_zone: str = pt.Field(
        dtype=pl.String, description="ASHRAE/IECC 2004 climate zone"
    )
    sqft: float = pt.Field(dtype=pl.Float64, description="modeled floor area (ft^2)")
    weight: float = pt.Field(
        dtype=pl.Float64,
        description="state-level expansion weight -- real dwellings this model "
        "represents; multiply-and-sum to scale the sample to state totals "
        "(uniform per state; not a percentage)",
    )
    annual_electricity_total_kwh: float = pt.Field(
        dtype=pl.Float64, description="annual total electricity (kWh)"
    )
    annual_electricity_cooling_kwh: float = pt.Field(
        dtype=pl.Float64,
        description="annual cooling electricity (kWh); 0 for non-electric cooling",
    )
    annual_electricity_heating_kwh: float = pt.Field(
        dtype=pl.Float64,
        description="annual electric-heating electricity (kWh); 0 for fossil heating",
    )


# --------------------------------------------------------------------------- #
# Parse -> bronze table
# --------------------------------------------------------------------------- #


def parquet_to_metadata_table(data: bytes) -> pl.DataFrame:
    """
    Parse a raw state metadata parquet into the typed bronze table (one row per
    building).
    """
    frame = oedi.read_curated(data, METADATA_RAW_TO_COLUMN, "metadata")
    return oedi.shape_to_schema(
        frame, ResstockMetadataBronzeSchema, METADATA_COLUMNS, METADATA_KEY_COLUMNS
    )


def metadata_for_puma(table: pl.DataFrame, puma_gisjoin: str) -> pl.DataFrame:
    """
    One PUMA's rows of a state metadata table -- what a metadata write holds.

    Fails loud on an empty result rather than writing an empty dataset under a PUMA's
    key: a key resolving to no buildings is worse than a missing one, since a later
    run finds it covered and never looks again. An empty slice means a PUMA that is
    not in this state, or a 2020-vintage code against a 2010-vintage release.

    **Permanent**, so a caller holding a list can tell this apart from a blip and
    drop the one PUMA instead of the run: the state file has been read, and it has no
    such code to find on a second look. ``PermanentFetchError`` subclasses
    ``PipelineValueError``, so a caller that only wants the loud failure still gets
    it unchanged.

    Raises:
        failures.PermanentFetchError: If the PUMA has no buildings in this table.
    """
    slice_ = table.filter(pl.col("puma_gisjoin") == puma_gisjoin)
    if slice_.is_empty():
        msg = (
            f"no buildings found for PUMA {puma_gisjoin} in this metadata table; "
            "check the PUMA GISJOIN and state pairing"
        )
        raise failures.PermanentFetchError(msg)
    return slice_


# --------------------------------------------------------------------------- #
# Write -> permanent storage
# --------------------------------------------------------------------------- #


def write_metadata_bronze(
    table: pl.DataFrame,
    params: ResstockMetadataRequestArgs,
    root_uri: str | Path,
    writer: str = DEFAULT_WRITER,
    write_time: dt.datetime | None = None,
) -> ManifestRow:
    """
    Write the metadata bronze frame through the shared catalog machinery
    """
    return columnar.write_dataset(
        table,
        ResstockMetadataBronzeSchema,
        METADATA_DATASET_NAME,
        params,
        str(root_uri),
        writer=writer,
        write_time=write_time,
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def ingest_metadata_bronze(
    args: ResstockMetadataRequestArgs | None = None,
    root_uri: str | Path = DEFAULT_BASE_DIR,
    writer: str = DEFAULT_WRITER,
    client: httpx.Client | None = None,
    write_time: dt.datetime | None = None,
) -> ManifestRow:
    """
    Full metadata bronze step for one PUMA: fetch the state parquet, parse it, keep
    that PUMA's buildings, write bronze
    """
    if args is None:
        args = ResstockMetadataRequestArgs()
    rows, rejected = ingest_metadata_bronze_for_pumas(
        [args], root_uri=root_uri, writer=writer, client=client, write_time=write_time
    )
    if not rows:
        # One PUMA asked for and none written: there is nothing to be tolerant
        # *about*, so the single-PUMA door keeps raising. Tolerance is only
        # meaningful across a list, and only its caller knows there is a list.
        raise failures.PermanentFetchError(rejected[0]["error"])
    return rows[0]


def ingest_metadata_bronze_for_pumas(
    args: Sequence[ResstockMetadataRequestArgs],
    root_uri: str | Path = DEFAULT_BASE_DIR,
    writer: str = DEFAULT_WRITER,
    client: httpx.Client | None = None,
    write_time: dt.datetime | None = None,
) -> tuple[list[ManifestRow], list[dict[str, str]]]:
    """
    Several PUMAs of one state metadata file, on **one** download.

    The write is per PUMA but the source file is per state, so the fetch is hoisted
    here rather than repeated per key: five PUMAs of a state pull 54 MB once, not five
    times. :func:`ingest_metadata_bronze` is the single-PUMA case of this.

    Every request must name the same state and release -- that is what makes one
    download serve them all, so a mixed list is a caller error rather than something
    to paper over with a second fetch.

    Returns ``(rows, rejected)``: a write per PUMA the file has buildings for, and a
    :func:`failures.rejected` record per PUMA it does not. A code the file cannot
    answer for is **not** allowed to cost its neighbours their write -- the download
    is already paid for and the other PUMAs are sitting in the frame, so failing the
    call would throw away work that has already succeeded. The caller decides what an
    empty ``rows`` means; :func:`ingest_metadata_bronze`, which asks about exactly
    one PUMA, still raises.

    Raises:
        PipelineValueError: If *args* is empty or spans more than one source file --
            both caller errors, unlike a PUMA the release simply does not have.
    """
    if not args:
        msg = "no metadata requests given; nothing to ingest"
        raise common.exceptions.PipelineValueError(msg)
    urls = {request.object_url() for request in args}
    if len(urls) != 1:
        msg = (
            f"metadata requests span {len(urls)} source files; one call reads one "
            "state's file, so group the requests by state first"
        )
        raise common.exceptions.PipelineValueError(msg)
    root_uri = oedi.resolve_root_uri(root_uri)

    data = download_object(urls.pop(), client=client)
    table = parquet_to_metadata_table(data)
    logger.info(
        "metadata bronze: %d building(s) in state %s, for %d PUMA(s)",
        table.height,
        args[0].state,
        len(args),
    )

    rows: list[ManifestRow] = []
    rejected: list[dict[str, str]] = []
    for request in args:
        try:
            puma_table = metadata_for_puma(table, request.puma_gisjoin)
        except failures.PermanentFetchError as exc:
            # The file has been read and this code is not in it. Record it and carry
            # on down the list rather than abandoning the PUMAs that are.
            logger.warning(
                "metadata bronze: state %s has no PUMA %s; skipping it and "
                "continuing with the rest of the request",
                request.state,
                request.puma_gisjoin,
            )
            rejected.append(
                failures.rejected(request.puma_gisjoin, "resstock", str(exc))
            )
            continue
        row = write_metadata_bronze(
            puma_table,
            params=request,
            root_uri=root_uri,
            writer=writer,
            write_time=write_time,
        )
        logger.info(
            "metadata bronze: wrote %d building(s) for PUMA %s -> %s",
            puma_table.height,
            request.puma_gisjoin,
            row.data_uri,
        )
        rows.append(row)
    return rows, rejected


# =========================================================================== #
# PUMA-batched timeseries (assemble every building in a PUMA into one dataset)
# =========================================================================== #

# Resolve every building in a PUMA from the metadata file, fetch their individual
# timeseries concurrently, and store them together keyed by (timestamp, bldg_id).

DEFAULT_PUMA = oedi.DEFAULT_PUMA


class ResstockPumaTimeseriesRequestArgs(ResstockRequestBase):
    """
    Arguments describing a whole-PUMA timeseries request

    The dataset covers every building in ``puma_gisjoin`` for the state, so
    ``bldg_id`` is a join-key column inside the data rather than a request field.
    """

    puma_gisjoin: str = pydantic.Field(
        default=DEFAULT_PUMA,
        pattern=oedi.PUMA_PATTERN,
        description="NHGIS PUMA GISJOIN to assemble all buildings for",
    )

    def state_metadata_url(self) -> str:
        """
        Full HTTPS URL of the state metadata parquet used to resolve the PUMA's
        building ids
        """
        return (
            f"{BASE_URL}/{OEDI_PREFIX}/{self.release_year}/{self.source_dataset}/"
            f"metadata_and_annual_results/by_state/full/parquet/"
            f"state={self.state}/{self.state}_upgrade{self.upgrade}.parquet"
        )

    def building_timeseries_url(self, bldg_id: int) -> str:
        """
        Full HTTPS URL of a single building's timeseries parquet
        """
        return oedi.building_timeseries_url(self, bldg_id)


def resolve_puma_building_ids(
    args: ResstockPumaTimeseriesRequestArgs,
    client: httpx.Client | None = None,
) -> list[int]:
    """
    Read the state metadata file and return the sorted ``bldg_id`` list for the
    requested PUMA

    Only the two columns the filter needs are read. Fails loud if the PUMA has no
    buildings in that state (a bad PUMA / state pair).
    """
    data = download_object(args.state_metadata_url(), client=client)
    frame = pl.read_parquet(io.BytesIO(data), columns=["bldg_id", _METADATA_PUMA_RAW])
    ids = (
        frame.filter(pl.col(_METADATA_PUMA_RAW) == args.puma_gisjoin)["bldg_id"]
        .unique()
        .sort()
        .to_list()
    )
    if not ids:
        msg = (
            f"no buildings found for PUMA {args.puma_gisjoin} in state {args.state}; "
            "check the PUMA GISJOIN and state pairing"
        )
        raise failures.PermanentFetchError(msg)
    return [int(b) for b in ids]


def fetch_building_frame(
    args: ResstockPumaTimeseriesRequestArgs,
    bldg_id: int,
    client: httpx.Client,
) -> pl.DataFrame:
    """
    Fetch one building's curated timeseries -- the unit the orchestrator maps.

    Takes ``args`` first so a caller can bind it and map over ids.
    """
    return oedi.fetch_building_frame(bldg_id, args, client, TIMESERIES_RAW_TO_COLUMN)


def fetch_puma_timeseries_table(
    args: ResstockPumaTimeseriesRequestArgs,
    client: httpx.Client,
    bldg_ids: list[int],
) -> pl.DataFrame:
    """
    Fetch every building's timeseries concurrently and concat into the typed
    PUMA bronze table (delegates to the shared assembler with ResStock's raw map)
    """
    return oedi.fetch_puma_timeseries_table(
        args,
        client,
        bldg_ids,
        timeseries_raw_map=TIMESERIES_RAW_TO_COLUMN,
    )


def write_puma_timeseries_bronze(
    table: pl.DataFrame,
    params: ResstockPumaTimeseriesRequestArgs,
    root_uri: str | Path,
    writer: str = DEFAULT_WRITER,
    write_time: dt.datetime | None = None,
) -> ManifestRow:
    """
    Write the PUMA timeseries bronze frame through the shared catalog machinery
    """
    return columnar.write_dataset(
        table,
        ResstockTimeseriesBronzeSchema,
        TIMESERIES_DATASET_NAME,
        params,
        str(root_uri),
        writer=writer,
        write_time=write_time,
    )


def read_metadata_bronze(
    args: ResstockMetadataRequestArgs,
    root_uri: str | Path = DEFAULT_BASE_DIR,
    as_of: dt.datetime | None = None,
) -> pl.DataFrame:
    """Read a previously written metadata bronze table: one PUMA's buildings."""
    return columnar.read_dataset(
        ResstockMetadataBronzeSchema,
        METADATA_DATASET_NAME,
        args,
        oedi.resolve_root_uri(root_uri),
        as_of=as_of,
    )


def building_ids(table: pl.DataFrame, puma_gisjoin: str) -> list[int]:
    """
    Sorted ``bldg_id`` list for one PUMA, from a metadata bronze table.

    The filter is a no-op on a well-formed table and kept as the check that the rows
    a key resolved to are the rows it claims: a metadata write holding another PUMA's
    buildings would otherwise fan out a thousand fetches for the wrong ones.

    Taking the ids from the metadata bronze the flow just wrote is what stops the
    state file (~54 MB for CA) being downloaded twice per ingest.

    Raises:
        PipelineValueError: If the PUMA has no buildings in this table.
    """
    ids = sorted(
        int(b)
        for b in table.filter(pl.col("puma_gisjoin") == puma_gisjoin)["bldg_id"]
        .unique()
        .to_list()
    )
    if not ids:
        msg = (
            f"no buildings found for PUMA {puma_gisjoin} in this metadata table; "
            "check the PUMA GISJOIN and state pairing"
        )
        raise common.exceptions.PipelineValueError(msg)
    return ids


def read_puma_timeseries_bronze(
    args: ResstockPumaTimeseriesRequestArgs,
    root_uri: str | Path = DEFAULT_BASE_DIR,
    as_of: dt.datetime | None = None,
    expected_buildings: int | None = None,
    require_complete: bool = False,
) -> pl.DataFrame:
    """
    Read a whole PUMA back: one manifest scan, then the write it resolves to.

    A PUMA is one entry keyed by exactly these args, so this is a key a caller can
    build. The read scans and matches on a subset rather than resolving exactly, only
    for the sake of writes already on disk -- see
    :func:`~...oedi_building_stock.read_puma_timeseries`.

    Pass ``expected_buildings`` -- the PUMA's count, from its metadata bronze -- to
    have a shortfall reported rather than silently summed. It is the only way a short
    PUMA is visible: the row count looks healthy either way.
    """
    return oedi.read_puma_timeseries(
        ResstockTimeseriesBronzeSchema,
        TIMESERIES_DATASET_NAME,
        args,
        root_uri,
        as_of=as_of,
        expected_buildings=expected_buildings,
        require_complete=require_complete,
    )


# The source-specific half of the shared PUMA ingest: which dataset it writes, and
# the four functions that differ. Defined here, after the functions it names.
_TIMESERIES_SOURCE = oedi.PumaTimeseriesSource(
    dataset_name=TIMESERIES_DATASET_NAME,
    resolve_ids=resolve_puma_building_ids,
    fetch_table=fetch_puma_timeseries_table,
    write_table=write_puma_timeseries_bronze,
)


def ingest_puma_timeseries_bronze(
    args: ResstockPumaTimeseriesRequestArgs | None = None,
    root_uri: str | Path = DEFAULT_BASE_DIR,
    writer: str = DEFAULT_WRITER,
    client: httpx.Client | None = None,
    write_time: dt.datetime | None = None,
    bldg_ids: list[int] | None = None,
    force_refresh: bool = False,
    fetch_buildings: oedi.BuildingFetcher | None = None,
    skip_bldg_ids: set[int] | None = None,
) -> list[ManifestRow]:
    """
    Full PUMA timeseries bronze step for this source -- see
    :func:`~external_data.load_pipeline.oedi_building_stock.ingest_puma_timeseries`,
    which both sources share.

    A named function rather than an alias, so this source's defaults (its writer, its
    request class) stay where a reader of this module looks for them.
    """
    return oedi.ingest_puma_timeseries(
        _TIMESERIES_SOURCE,
        args if args is not None else ResstockPumaTimeseriesRequestArgs(),
        root_uri=root_uri,
        writer=writer,
        client=client,
        write_time=write_time,
        bldg_ids=bldg_ids,
        force_refresh=force_refresh,
        fetch_buildings=fetch_buildings,
        skip_bldg_ids=skip_bldg_ids,
    )


if __name__ == "__main__":
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    metadata_row = ingest_metadata_bronze()
    print(metadata_row.data_uri)
    for puma_row in ingest_puma_timeseries_bronze():
        print(puma_row.data_uri)
