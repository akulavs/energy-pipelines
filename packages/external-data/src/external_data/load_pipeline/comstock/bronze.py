"""
ComStock (commercial building stock) -- bronze ingestion

Two bronze datasets, mirroring ResStock's pair so the two sources read alike:
one row per building of metadata, and one row per building x timestep of load,
joined on ``bldg_id``.

- ``comstock_puma_metadata_bronze`` -- one row per building model for a PUMA, from
  the ``metadata_and_annual_results_aggregates/by_state_and_puma`` family. Despite
  the family name this is not a rollup: it is per-building metadata with the
  census-tract duplication collapsed, so a PUMA's building ids and expansion
  weights come from **one** request.
- ``comstock_timeseries_bronze`` -- one row per building x 15-min timestep: the
  unweighted per-model electricity load profile. Building ids come from the PUMA
  metadata above; their ``timeseries_individual_buildings`` files are fetched per
  building and written as one unpartitioned file per PUMA.

The release publishes more -- per-county metadata, pre-weighted
``timeseries_aggregates``, weather, PUMA boundary geometry -- and none of it is
ingested here. The aggregate path is far cheaper (~15 requests against ~950) but
carries no ``bldg_id``, so it answers "what does this PUMA draw" and never "which
buildings drew it", and an aggregate cannot be un-summed.
"""

from __future__ import annotations
import datetime as dt
import io
import logging
from pathlib import Path
import httpx
import patito as pt
import polars as pl
import pydantic
import common.exceptions
from common.frames import BaseDataFrameSchema
from common.storage import columnar
from common.storage.manifest import ManifestRow
from external_data.load_pipeline import oedi_building_stock as oedi
from external_data.load_pipeline.oedi_building_stock import download_object

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Shared OEDI location constants re-exported for the module's public/test surface
BASE_URL = oedi.BASE_URL
OEDI_PREFIX = oedi.OEDI_PREFIX

# Latest ComStock release (Nov 2025)
RELEASE = "comstock_amy2018_release_3"
DATASET = RELEASE

DEFAULT_WRITER = "comstock_bronze"
DEFAULT_BASE_DIR = oedi.DEFAULT_BASE_DIR

DEFAULT_PUMA = oedi.DEFAULT_PUMA


# --------------------------------------------------------------------------- #
# Request
# --------------------------------------------------------------------------- #


class ComstockRequestBase(oedi.OediBuildingStockRequestBase):
    """
    Shared release + geography fields for ComStock OEDI requests
    """

    source_dataset: str = pydantic.Field(
        default=DATASET,
        description="OEDI ComStock release directory the data was retrieved from",
    )


class ComstockPumaMetadataRequestArgs(ComstockRequestBase):
    """
    Arguments describing one PUMA's building-level metadata request

    From the ``metadata_and_annual_results_aggregates/by_state_and_puma`` family.
    "Aggregates" describes how the *rows* were combined, not the values: a building
    appears once, carrying the ``weight`` the per-county file splits across every
    census tract it represents. Measured on DC PUMA G11000101 -- 946 rows here
    against 2,985 tract-duplicated rows there for the same 946 buildings, with
    per-building ``weight`` equal to the sum of the split weights.
    """

    puma_gisjoin: str = pydantic.Field(
        default=DEFAULT_PUMA,
        pattern=oedi.PUMA_PATTERN,
        description="NHGIS PUMA GISJOIN whose buildings the file describes",
    )

    def object_url(self) -> str:
        """
        Full HTTPS URL of the PUMA's metadata parquet on the OEDI data lake
        """
        return (
            f"{BASE_URL}/{OEDI_PREFIX}/{self.release_year}/{self.source_dataset}/"
            f"metadata_and_annual_results_aggregates/by_state_and_puma/full/parquet/"
            f"state={self.state}/puma={self.puma_gisjoin}/"
            f"{self.state}_{self.puma_gisjoin}_upgrade{self.upgrade}_agg.parquet"
        )


class ComstockPumaTimeseriesRequestArgs(ComstockRequestBase):
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

    def building_timeseries_url(self, bldg_id: int) -> str:
        """
        Full HTTPS URL of a single building's timeseries parquet
        """
        return oedi.building_timeseries_url(self, bldg_id)


# --------------------------------------------------------------------------- #
# Dataset schemas (catalog / manifest)
# --------------------------------------------------------------------------- #


# Curated timeseries columns. ComStock timeseries drops the ``..kwh`` suffix the
# metadata file carries.
TIMESERIES_RAW_TO_COLUMN: dict[str, str] = {
    "timestamp": "timestamp",
    "bldg_id": "bldg_id",
    "out.electricity.total.energy_consumption": "electricity_total_kwh",
    "out.electricity.cooling.energy_consumption": "electricity_cooling_kwh",
    "out.electricity.heating.energy_consumption": "electricity_heating_kwh",
}

# The timeseries bronze schema + column lists are shared
TIMESERIES_DATASET_NAME = "comstock_timeseries_bronze"
ComstockTimeseriesBronzeSchema = oedi.OediBuildingStockTimeseriesSchema
TIMESERIES_COLUMNS = oedi.TIMESERIES_COLUMNS
TIMESERIES_KEY_COLUMNS = oedi.TIMESERIES_KEY_COLUMNS


PUMA_METADATA_DATASET_NAME = "comstock_puma_metadata_bronze"

# Raw OEDI column -> curated bronze field, for the per-PUMA metadata file.
#
# No county / tract / climate-zone columns, though the file has them: every one is
# prefixed ``in.as_simulated_`` -- where the model was *simulated*, not where its
# load sits. For DC PUMA G11000101 the simulated counties include two in Virginia,
# so a column named ``county_gisjoin`` would hold another state's county. The
# location stays the PUMA the file is keyed by.
PUMA_METADATA_RAW_TO_COLUMN: dict[str, str] = {
    "bldg_id": "bldg_id",
    "upgrade": "upgrade",
    "in.state": "state",
    "in.nhgis_puma_gisjoin": "puma_gisjoin",
    "in.comstock_building_type": "building_type",
    "in.vintage": "vintage",
    "in.hvac_system_type": "hvac_system_type",
    "in.heating_fuel": "heating_fuel",
    "in.sqft..ft2": "sqft",
    "calc.weighted.sqft..ft2": "weighted_sqft",
    "weight": "weight",
    "out.electricity.total.energy_consumption..kwh": "annual_electricity_total_kwh",
    "out.electricity.cooling.energy_consumption..kwh": "annual_electricity_cooling_kwh",
    "out.electricity.heating.energy_consumption..kwh": "annual_electricity_heating_kwh",
}
PUMA_METADATA_COLUMNS = list(PUMA_METADATA_RAW_TO_COLUMN.values())
PUMA_METADATA_KEY_COLUMNS = ["bldg_id"]
# No partition columns: this write is already one PUMA, so the directory would
# hold a single file and repeat the key. The manifest is the index.


class ComstockPumaMetadataBronzeSchema(BaseDataFrameSchema):
    """
    One row per building model in a PUMA -- the tract duplication collapsed.

    The per-county metadata file lists a building once per census tract it represents;
    this file lists it once, which is what makes it the table to join a timeseries to.
    Joining the tract-duplicated one instead fans each profile out by its tract count
    -- measured at 6.2x on average within a PUMA, and 1,147 rows for one California
    building.

    Annual totals are Float64 while the per-interval timeseries columns are Float32:
    35,040 intervals a year get summed, so the annual figures are the exact ones.
    """

    bldg_id: int = pt.Field(
        dtype=pl.Int64,
        description="ComStock building model id; unique in this table, and the "
        "join key to comstock_timeseries_bronze",
    )
    upgrade: int = pt.Field(
        dtype=pl.Int32, description="upgrade / measure-package id (0 = baseline)"
    )
    state: str = pt.Field(dtype=pl.String, description="two-letter state code")
    puma_gisjoin: str = pt.Field(
        dtype=pl.String, description="NHGIS PUMA GISJOIN the buildings belong to"
    )
    building_type: str = pt.Field(
        dtype=pl.String,
        description="ComStock commercial building type (CamelCase enum)",
    )
    vintage: str = pt.Field(dtype=pl.String, description="building age cohort")
    hvac_system_type: str = pt.Field(
        dtype=pl.String, description="commercial HVAC system type"
    )
    heating_fuel: str = pt.Field(
        dtype=pl.String, description="primary heating fuel (CamelCase enum)"
    )
    sqft: float = pt.Field(dtype=pl.Float64, description="modeled floor area (ft^2)")
    weighted_sqft: float = pt.Field(
        dtype=pl.Float64,
        description="floor area weighted to real-world scale (ft^2); equals "
        "sqft x weight",
    )
    weight: float = pt.Field(
        dtype=pl.Float64,
        description="expansion weight: how many real buildings this model stands "
        "in for **within this PUMA**. The source's per-county file splits a "
        "building's weight across every census tract it represents; this file sums "
        "the tracts of this PUMA only. So it is not the model's global weight -- "
        "measured on CA PUMA G06009703, 844 of 864 buildings represent more stock "
        "outside it than in -- but it is exactly the share this PUMA's load should "
        "reflect. Multiply and sum for a real-world total.",
    )
    annual_electricity_total_kwh: float = pt.Field(
        dtype=pl.Float64, description="annual total electricity (kWh), unweighted"
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


def parquet_to_puma_metadata_table(data: bytes) -> pl.DataFrame:
    """
    Parse a raw per-PUMA metadata parquet into the typed bronze table

    One row per building: unlike the per-county file there is nothing to de-duplicate,
    which is why this path exists.
    """
    frame = oedi.read_curated(data, PUMA_METADATA_RAW_TO_COLUMN, "PUMA metadata")
    return oedi.shape_to_schema(
        frame,
        ComstockPumaMetadataBronzeSchema,
        PUMA_METADATA_COLUMNS,
        PUMA_METADATA_KEY_COLUMNS,
    )


# --------------------------------------------------------------------------- #
# Resolve PUMA -> building ids (from the per-PUMA metadata file)
# --------------------------------------------------------------------------- #


def puma_metadata_request(
    args: ComstockPumaTimeseriesRequestArgs,
) -> ComstockPumaMetadataRequestArgs:
    """The PUMA metadata key covering the same release/upgrade/geography."""
    return ComstockPumaMetadataRequestArgs(
        release_year=args.release_year,
        source_dataset=args.source_dataset,
        upgrade=args.upgrade,
        state=args.state,
        puma_gisjoin=args.puma_gisjoin,
    )


def building_ids(table: pl.DataFrame) -> list[int]:
    """
    Sorted ``bldg_id`` list from a PUMA metadata table.

    Sorted so any slice of it is reproducible, and de-duplicated defensively: this
    table is one row per building, so a repeat would mean the source changed shape.
    """
    return sorted({int(b) for b in table["bldg_id"].to_list()})


def resolve_puma_building_ids(
    args: ComstockPumaTimeseriesRequestArgs,
    client: httpx.Client | None = None,
) -> list[int]:
    """
    Return the sorted ``bldg_id`` list for the requested PUMA, in **one** request.

    Reads the per-PUMA metadata file, which is keyed by exactly the geography being
    asked about -- the per-county files carry no PUMA index, so resolving from those
    meant downloading every county file in the state (58 for CA) per PUMA ingested.

    Prefer :func:`building_ids` when the metadata bronze is already written: the ids
    are then a local parquet read rather than a second download.

    Fails loud on an empty result, which means a bad PUMA / state pairing.
    """
    data = download_object(puma_metadata_request(args).object_url(), client=client)
    frame = pl.read_parquet(io.BytesIO(data), columns=["bldg_id"])
    ids = building_ids(frame)
    if not ids:
        msg = (
            f"no buildings found for PUMA {args.puma_gisjoin} in state {args.state}; "
            "check the PUMA GISJOIN and state pairing"
        )
        raise common.exceptions.PipelineValueError(msg)
    return ids


# --------------------------------------------------------------------------- #
# Fetch + assemble -> timeseries bronze table
# --------------------------------------------------------------------------- #


def fetch_building_frame(
    args: ComstockPumaTimeseriesRequestArgs,
    bldg_id: int,
    client: httpx.Client,
) -> pl.DataFrame:
    """
    Fetch one building's curated timeseries -- the unit the orchestrator maps.

    Takes ``args`` first so a caller can bind it and map over ids.
    """
    return oedi.fetch_building_frame(bldg_id, args, client, TIMESERIES_RAW_TO_COLUMN)


def fetch_puma_timeseries_table(
    args: ComstockPumaTimeseriesRequestArgs,
    client: httpx.Client,
    bldg_ids: list[int],
) -> pl.DataFrame:
    """
    Fetch every building's timeseries concurrently and concat into the typed
    PUMA bronze table (delegates to the shared assembler with ComStock's raw map)
    """
    return oedi.fetch_puma_timeseries_table(
        args,
        client,
        bldg_ids,
        timeseries_raw_map=TIMESERIES_RAW_TO_COLUMN,
    )


# --------------------------------------------------------------------------- #
# Write -> permanent storage
# --------------------------------------------------------------------------- #


def write_puma_metadata_bronze(
    table: pl.DataFrame,
    params: ComstockPumaMetadataRequestArgs,
    root_uri: str | Path,
    writer: str = DEFAULT_WRITER,
    write_time: dt.datetime | None = None,
) -> ManifestRow:
    """
    Write the per-PUMA metadata bronze frame through the shared catalog machinery
    """
    return columnar.write_dataset(
        table,
        ComstockPumaMetadataBronzeSchema,
        PUMA_METADATA_DATASET_NAME,
        params,
        str(root_uri),
        writer=writer,
        write_time=write_time,
    )


def write_puma_timeseries_bronze(
    table: pl.DataFrame,
    params: ComstockPumaTimeseriesRequestArgs,
    root_uri: str | Path,
    writer: str = DEFAULT_WRITER,
    write_time: dt.datetime | None = None,
) -> ManifestRow:
    """
    Write the PUMA timeseries bronze frame through the shared catalog machinery
    """
    return columnar.write_dataset(
        table,
        ComstockTimeseriesBronzeSchema,
        TIMESERIES_DATASET_NAME,
        params,
        str(root_uri),
        writer=writer,
        write_time=write_time,
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def ingest_puma_metadata_bronze(
    args: ComstockPumaMetadataRequestArgs | None = None,
    root_uri: str | Path = DEFAULT_BASE_DIR,
    writer: str = DEFAULT_WRITER,
    client: httpx.Client | None = None,
    write_time: dt.datetime | None = None,
) -> ManifestRow:
    """
    Full per-PUMA metadata bronze step: fetch the parquet, parse it, write bronze

    One request for the whole PUMA. The file carries ~1,300 columns and the curated
    set is 14 of them, so this is the pipeline's strongest case for pushing the
    projection to the source rather than downloading the file whole.
    """
    if args is None:
        args = ComstockPumaMetadataRequestArgs()
    root_uri = oedi.resolve_root_uri(root_uri)

    data = download_object(args.object_url(), client=client)
    table = parquet_to_puma_metadata_table(data)
    logger.info(
        "PUMA metadata bronze: %d building(s) for PUMA %s",
        table.height,
        args.puma_gisjoin,
    )

    row = write_puma_metadata_bronze(
        table, params=args, root_uri=root_uri, writer=writer, write_time=write_time
    )
    logger.info(
        "PUMA metadata bronze: wrote %s -> %s", PUMA_METADATA_DATASET_NAME, row.data_uri
    )
    return row


def read_puma_metadata_bronze(
    args: ComstockPumaMetadataRequestArgs,
    root_uri: str | Path = DEFAULT_BASE_DIR,
    as_of: dt.datetime | None = None,
) -> pl.DataFrame:
    """
    Read a previously written per-PUMA metadata bronze table -- how a caller gets
    building ids and weights without paying for a second download.
    """
    return columnar.read_dataset(
        ComstockPumaMetadataBronzeSchema,
        PUMA_METADATA_DATASET_NAME,
        args,
        oedi.resolve_root_uri(root_uri),
        as_of=as_of,
    )


def read_puma_timeseries_bronze(
    args: ComstockPumaTimeseriesRequestArgs,
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
        ComstockTimeseriesBronzeSchema,
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
    args: ComstockPumaTimeseriesRequestArgs | None = None,
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
        args if args is not None else ComstockPumaTimeseriesRequestArgs(),
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

    print(ingest_puma_metadata_bronze().data_uri)
    for puma_row in ingest_puma_timeseries_bronze():
        print(puma_row.data_uri)
