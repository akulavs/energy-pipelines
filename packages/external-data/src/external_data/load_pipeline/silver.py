"""
dsgrid industrial -- silver union.

dsgrid publishes two source files, each with a metadata and a timeseries bronze
table. This layer stacks them **by kind**, so four bronze tables become two
silver ones:

    dsgrid_industrial_metadata_bronze        ┐
    dsgrid_industrial_gaps_metadata_bronze   ┘-> dsgrid_industrial_metadata_silver

    dsgrid_industrial_timeseries_bronze      ┐
    dsgrid_industrial_gaps_timeseries_bronze ┘-> dsgrid_industrial_timeseries_silver

A **union**, not a horizontal join. The two sources describe *different sectors of
the same counties* at different granularity -- 4-digit NAICS subsectors against
2-digit sectors -- so there is no key to join on, and joining on county alone would
fan each gaps row across that county's subsectors and multiply its energy.

Two columns make the stack work:

- ``naics_code`` -- the harmonised sector id (``naics_subsector`` from the
  manufacturing file, ``naics_sector`` from gaps). One is 4 digits and the other 2,
  so they cannot collide.
- ``source`` -- which file a row came from, so manufacturing and non-manufacturing
  stay separable after stacking.

The manufacturing file decomposes its load into 12 electricity end uses; gaps carries
a single un-decomposed one, so gaps rows leave those 12 columns **null**. What keeps
the total recoverable is ``electricity_total_mwh`` (and its annual counterpart),
populated on every row: summing it over a county x hour gives the whole industrial
load, both halves together.

The axis stays dsgrid's own -- hourly, the 2012 modelled year, one national clock
(-05:00) for every county -- because this conforms bronze rather than re-basing it.
``schema.DSGRID_UTC_OFFSET_MINUTES`` and ``schema.DSGRID_YEAR`` are what a consumer
needs to align it against the building stock later.

This module also derives every bronze manifest key from a geography plus a source's
fetch config, which the ingest flows use.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import patito as pt
import polars as pl
import pydantic

import common.exceptions
from common.frames import BaseDataFrameSchema
from common.storage import columnar
from common.storage.manifest import ManifestRow
from external_data.load_pipeline import schema
from external_data.load_pipeline.comstock import bronze as comstock_bronze
from external_data.load_pipeline.dsgrid import bronze as dsgrid_bronze
from external_data.load_pipeline.resstock import bronze as resstock_bronze

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

METADATA_DATASET_NAME = "dsgrid_industrial_metadata_silver"
TIMESERIES_DATASET_NAME = "dsgrid_industrial_timeseries_silver"

DEFAULT_WRITER = "dsgrid_industrial_silver"

# Medallion roots (siblings): silver reads bronze from the bronze root and writes
# the silver datasets to the silver root.
DEFAULT_BRONZE_DIR = "bronze"
DEFAULT_SILVER_DIR = "silver"

# Harmonised column names the two sources are stacked under.
SECTOR_COLUMN = "naics_code"
SECTOR_NAME_COLUMN = "sector_name"
SOURCE_COLUMN = "source"

METADATA_KEY_COLUMNS = ["county_gisjoin", SECTOR_COLUMN]
TIMESERIES_KEY_COLUMNS = ["timestamp", "county_gisjoin", SECTOR_COLUMN]

# No partition columns. A write is one state, so a ``state=XX/`` directory would hold
# a single file and repeat what the key says; partitioning the timeseries by
# ``source`` would split the very thing this layer unified, and a reader wanting one
# half filters a column already in the file. The manifest is the index, as it is for
# every other dataset here.


# --------------------------------------------------------------------------- #
# Schemas (catalog / manifest)
# --------------------------------------------------------------------------- #


class DsgridIndustrialMetadataSilverSchema(BaseDataFrameSchema):
    """
    One row per county x NAICS code: both dsgrid sources' annual totals, stacked.

    The 12 end-use breakout columns are **null on non-manufacturing rows** -- the
    gaps file publishes a single un-decomposed end use, so there is nothing to put
    there. ``annual_electricity_total_mwh`` is populated for every row, which is
    what keeps the industrial total recoverable by summing.
    """

    state: str = pt.Field(dtype=pl.String, description="two-letter state code")
    county_gisjoin: str = pt.Field(
        dtype=pl.String, description="NHGIS county GISJOIN (part of the row key)"
    )
    county_fips: str = pt.Field(
        dtype=pl.String, description="5-digit county FIPS (source geography id)"
    )
    county_name: str = pt.Field(
        dtype=pl.String, description="county name, e.g. 'Autauga County, AL'"
    )
    source: str = pt.Field(
        dtype=pl.String,
        description="which dsgrid file the row came from: 'industrial' "
        "(manufacturing) or 'industrial_gaps' (agriculture, mining, construction). "
        "Keeps the two separable after stacking.",
    )
    naics_code: str = pt.Field(
        dtype=pl.String,
        description="harmonised NAICS identifier: the 4-digit subsector from the "
        "manufacturing file or the 2-digit sector from the gaps file. The digit "
        "counts differ, so the two can never collide.",
    )
    sector_name: str = pt.Field(
        dtype=pl.String, description="human-readable name of that NAICS code"
    )
    annual_electricity_total_mwh: float = pt.Field(
        dtype=pl.Float64,
        description="annual total electricity for the county x code (MWh). "
        "Populated for every row, both sources.",
    )
    annual_electricity_conventional_boiler_use_mwh: float | None = pt.Field(
        dtype=pl.Float64, description="annual conventional boiler electricity (MWh)"
    )
    annual_electricity_process_heating_mwh: float | None = pt.Field(
        dtype=pl.Float64, description="annual process-heating electricity (MWh)"
    )
    annual_electricity_process_cooling_and_refrigeration_mwh: float | None = pt.Field(
        dtype=pl.Float64, description="annual process cooling & refrigeration (MWh)"
    )
    annual_electricity_machine_drive_mwh: float | None = pt.Field(
        dtype=pl.Float64, description="annual machine-drive electricity (MWh)"
    )
    annual_electricity_electro_chemical_processes_mwh: float | None = pt.Field(
        dtype=pl.Float64,
        description="annual electro-chemical process electricity (MWh)",
    )
    annual_electricity_other_process_use_mwh: float | None = pt.Field(
        dtype=pl.Float64, description="annual other process-use electricity (MWh)"
    )
    annual_electricity_facility_hvac_mwh: float | None = pt.Field(
        dtype=pl.Float64, description="annual facility HVAC electricity (MWh)"
    )
    annual_electricity_facility_lighting_mwh: float | None = pt.Field(
        dtype=pl.Float64, description="annual facility lighting electricity (MWh)"
    )
    annual_electricity_other_facility_support_mwh: float | None = pt.Field(
        dtype=pl.Float64, description="annual other facility-support electricity (MWh)"
    )
    annual_electricity_onsite_transportation_mwh: float | None = pt.Field(
        dtype=pl.Float64, description="annual on-site transportation electricity (MWh)"
    )
    annual_electricity_other_nonprocess_use_mwh: float | None = pt.Field(
        dtype=pl.Float64, description="annual other non-process electricity (MWh)"
    )
    annual_electricity_end_use_not_reported_mwh: float | None = pt.Field(
        dtype=pl.Float64,
        description="annual electricity with end use not reported (MWh)",
    )


class DsgridIndustrialTimeseriesSilverSchema(BaseDataFrameSchema):
    """
    One row per county x NAICS code x hour: both dsgrid sources' hourly load,
    stacked.

    As with the metadata, the 12 end-use columns are **null on non-manufacturing
    rows** while ``electricity_total_mwh`` is populated throughout -- so summing
    that column over a county x hour gives the full industrial load.

    Values are MWh delivered over the hour, which *is* the average MW over it.
    Timestamps are dsgrid's own: interval-ending, 2012, one national clock.
    """

    timestamp: dt.datetime = pt.Field(
        dtype=pl.Datetime(time_unit="us"),
        description="interval-ENDING timestamp, as dsgrid publishes it (naive "
        "local standard, 2012 EST for every county)",
    )
    county_gisjoin: str = pt.Field(
        dtype=pl.String, description="NHGIS county GISJOIN the load belongs to"
    )
    state: str = pt.Field(dtype=pl.String, description="two-letter state code")
    source: str = pt.Field(
        dtype=pl.String,
        description="'industrial' (manufacturing) or 'industrial_gaps' "
        "(agriculture, mining, construction)",
    )
    naics_code: str = pt.Field(
        dtype=pl.String,
        description="harmonised NAICS identifier: 4-digit subsector from the "
        "manufacturing file, 2-digit sector from the gaps file",
    )
    electricity_total_mwh: float = pt.Field(
        dtype=pl.Float32,
        description="total electricity for the interval (MWh). Populated for "
        "every row, both sources -- sum it over a county x hour for the "
        "industrial total.",
    )
    electricity_conventional_boiler_use_mwh: float | None = pt.Field(
        dtype=pl.Float32, description="conventional boiler electricity (MWh)"
    )
    electricity_process_heating_mwh: float | None = pt.Field(
        dtype=pl.Float32, description="process-heating electricity (MWh)"
    )
    electricity_process_cooling_and_refrigeration_mwh: float | None = pt.Field(
        dtype=pl.Float32, description="process cooling & refrigeration (MWh)"
    )
    electricity_machine_drive_mwh: float | None = pt.Field(
        dtype=pl.Float32, description="machine-drive electricity (MWh)"
    )
    electricity_electro_chemical_processes_mwh: float | None = pt.Field(
        dtype=pl.Float32, description="electro-chemical process electricity (MWh)"
    )
    electricity_other_process_use_mwh: float | None = pt.Field(
        dtype=pl.Float32, description="other process-use electricity (MWh)"
    )
    electricity_facility_hvac_mwh: float | None = pt.Field(
        dtype=pl.Float32, description="facility HVAC electricity (MWh)"
    )
    electricity_facility_lighting_mwh: float | None = pt.Field(
        dtype=pl.Float32, description="facility lighting electricity (MWh)"
    )
    electricity_other_facility_support_mwh: float | None = pt.Field(
        dtype=pl.Float32, description="other facility-support electricity (MWh)"
    )
    electricity_onsite_transportation_mwh: float | None = pt.Field(
        dtype=pl.Float32, description="on-site transportation electricity (MWh)"
    )
    electricity_other_nonprocess_use_mwh: float | None = pt.Field(
        dtype=pl.Float32, description="other non-process electricity (MWh)"
    )
    electricity_end_use_not_reported_mwh: float | None = pt.Field(
        dtype=pl.Float32, description="electricity with end use not reported (MWh)"
    )


# --------------------------------------------------------------------------- #
# Bronze key derivation
# --------------------------------------------------------------------------- #

# Each helper combines a geography with one source's fetch config to produce that
# bronze dataset's manifest key. The ingest flows use these; the union below uses
# only the dsgrid ones.


def resstock_timeseries_request(
    geography: schema.LoadGeography,
    resstock: schema.ResstockRequestArgs | None = None,
) -> resstock_bronze.ResstockPumaTimeseriesRequestArgs:
    """Derive the ResStock PUMA timeseries bronze key."""
    resstock = resstock or schema.ResstockRequestArgs()
    return resstock_bronze.ResstockPumaTimeseriesRequestArgs(
        source_dataset=resstock.source_dataset,
        release_year=resstock.release_year,
        upgrade=resstock.upgrade,
        state=geography.state,
        puma_gisjoin=geography.puma_gisjoin,
    )


def resstock_metadata_request(
    geography: schema.LoadGeography,
    resstock: schema.ResstockRequestArgs | None = None,
) -> resstock_bronze.ResstockMetadataRequestArgs:
    """
    Derive the ResStock metadata bronze key.

    Keyed by the PUMA like every other write here, even though OEDI publishes the
    file per state: the write holds that PUMA's buildings. The state stays in the key
    because it is what the download is addressed by, and one read of it serves however
    many of that state's PUMAs a run asked for.
    """
    resstock = resstock or schema.ResstockRequestArgs()
    return resstock_bronze.ResstockMetadataRequestArgs(
        source_dataset=resstock.source_dataset,
        release_year=resstock.release_year,
        upgrade=resstock.upgrade,
        state=geography.state,
        puma_gisjoin=geography.puma_gisjoin,
    )


def comstock_timeseries_request(
    geography: schema.LoadGeography,
    comstock: schema.ComstockRequestArgs | None = None,
) -> comstock_bronze.ComstockPumaTimeseriesRequestArgs:
    """Derive the ComStock PUMA timeseries bronze key."""
    comstock = comstock or schema.ComstockRequestArgs()
    return comstock_bronze.ComstockPumaTimeseriesRequestArgs(
        source_dataset=comstock.source_dataset,
        release_year=comstock.release_year,
        upgrade=comstock.upgrade,
        state=geography.state,
        puma_gisjoin=geography.puma_gisjoin,
    )


def comstock_puma_metadata_request(
    geography: schema.LoadGeography,
    comstock: schema.ComstockRequestArgs | None = None,
) -> comstock_bronze.ComstockPumaMetadataRequestArgs:
    """
    Derive the ComStock per-PUMA metadata bronze key.

    One file per PUMA, one row per building, ``weight`` already summed over the census
    tracts a model represents -- so building ids and weights come from here, not from
    the per-county table.
    """
    comstock = comstock or schema.ComstockRequestArgs()
    return comstock_bronze.ComstockPumaMetadataRequestArgs(
        source_dataset=comstock.source_dataset,
        release_year=comstock.release_year,
        upgrade=comstock.upgrade,
        state=geography.state,
        puma_gisjoin=geography.puma_gisjoin,
    )


def dsgrid_requests(
    state: str,
    dsgrid: schema.DsgridRequestArgs | None = None,
) -> list[dsgrid_bronze.DsgridRequestArgs]:
    """
    Derive the dsgrid bronze keys -- one per requested source file.

    Takes a bare ``state`` because dsgrid publishes per state and covers every
    county at once; the PUMA plays no part.
    """
    dsgrid = dsgrid or schema.DsgridRequestArgs()
    return [
        dsgrid_bronze.DsgridRequestArgs(
            source=source, source_dataset=dsgrid.source_dataset, state=state
        )
        for source in dsgrid.sources
    ]


# --------------------------------------------------------------------------- #
# Union
# --------------------------------------------------------------------------- #


def harmonise(frame: pl.DataFrame, source: dsgrid_bronze.DsgridSource) -> pl.DataFrame:
    """
    Rename one source's sector columns to the shared names and stamp its origin.

    ``naics_subsector`` / ``naics_sector`` both become ``naics_code`` and the name
    columns both become ``sector_name``, which is what lets two differently-grained
    tables stack at all. The names come from the source's own profile rather than a
    parameter -- two arguments that must agree is an invitation for them not to.

    The name column is renamed only when present, since this serves both kinds: only
    metadata carries it. The sector column is required on both, so a frame missing
    that still fails loudly.
    """
    profile = dsgrid_bronze.PROFILES[source]
    renames = {profile.sector_column: SECTOR_COLUMN}
    name_column = profile.sector_name_column
    if name_column in frame.columns and name_column != SECTOR_NAME_COLUMN:
        renames[name_column] = SECTOR_NAME_COLUMN
    return frame.rename(renames).with_columns(
        pl.lit(str(source), dtype=pl.String).alias(SOURCE_COLUMN)
    )


def union_sources(
    by_source: dict[dsgrid_bronze.DsgridSource, pl.DataFrame],
    schema_cls: type[BaseDataFrameSchema],
    key_columns: list[str],
) -> pl.DataFrame:
    """
    Stack the harmonised per-source frames into one table.

    A diagonal concat: the manufacturing file's 12 end-use columns do not exist on the
    gaps frame, so stacking leaves them null there rather than inventing a zero.
    ``electricity_total_mwh`` is on both, so the total stays summable across the table.
    """
    if not by_source:
        msg = "no dsgrid sources to union"
        raise common.exceptions.PipelineValueError(msg)

    parts = [harmonise(frame, source) for source, frame in by_source.items()]
    stacked = pl.concat(parts, how="diagonal_relaxed")

    missing = [c for c in schema_cls.columns if c not in stacked.columns]
    if missing:
        stacked = stacked.with_columns(
            *[pl.lit(None).alias(c) for c in missing],
        )
    stacked = _shape_to_schema(stacked, schema_cls, key_columns)

    # The sources are disjoint by NAICS grain (4-digit subsectors against 2-digit
    # sectors), so stacking cannot collide. Nothing else checks that invariant, and if
    # it broke -- a third source, or a re-grained file -- silver would emit duplicate
    # rows and every downstream sum would double-count in silence.
    duplicates = stacked.height - stacked.select(key_columns).n_unique()
    if duplicates:
        msg = (
            f"stacking produced {duplicates} duplicate row(s) on {key_columns}; "
            "the sources' NAICS codes are expected to be disjoint (4-digit "
            "subsectors vs 2-digit sectors)"
        )
        raise common.exceptions.PipelineValueError(msg)
    return stacked


def _shape_to_schema(
    frame: pl.DataFrame, schema_cls: type[BaseDataFrameSchema], key_columns: list[str]
) -> pl.DataFrame:
    """
    Cast to the silver schema dtypes, order/sort, and validate (the schema is the
    single source of dtype truth)
    """
    shaped = (
        schema_cls.DataFrame(frame.select(schema_cls.columns)).cast().sort(key_columns)
    )
    schema_cls.validate(shaped)
    return shaped


# --------------------------------------------------------------------------- #
# Read bronze -> silver tables
# --------------------------------------------------------------------------- #


def _resolve_root_uri(root_uri: str | Path) -> str:
    if "://" not in str(root_uri):
        return str(Path(root_uri).resolve())
    return str(root_uri)


def _read_bronze(
    schema_cls: type[BaseDataFrameSchema],
    dataset_name: str,
    params: pydantic.BaseModel,
    root_uri: str,
    label: str,
    as_of: dt.datetime | None = None,
) -> pl.DataFrame:
    try:
        return columnar.read_dataset(
            schema_cls, dataset_name, params, root_uri, as_of=as_of
        )
    except KeyError as exc:
        msg = (
            f"{label} bronze not found under {root_uri} for the requested state; "
            "ingest the bronze slice before building silver"
        )
        raise common.exceptions.PipelineValueError(msg) from exc


def build_metadata_table(
    request: schema.IndustrialLoadRequestArgs,
    bronze_root: str | Path,
    as_of: dt.datetime | None = None,
) -> pl.DataFrame:
    """Read every requested source's metadata bronze and stack them."""
    bronze_root = _resolve_root_uri(bronze_root)
    by_source = {
        params.source: _read_bronze(
            dsgrid_bronze.PROFILES[params.source].metadata_schema,
            dsgrid_bronze.PROFILES[params.source].metadata_dataset_name,
            params,
            bronze_root,
            f"dsgrid {params.source} metadata",
            as_of,
        )
        for params in dsgrid_requests(request.state, request.dsgrid)
    }
    table = union_sources(
        by_source, DsgridIndustrialMetadataSilverSchema, METADATA_KEY_COLUMNS
    )
    logger.info(
        "%s: %d row(s), %d county(ies), %d NAICS code(s) from %s",
        METADATA_DATASET_NAME,
        table.height,
        table["county_gisjoin"].n_unique(),
        table[SECTOR_COLUMN].n_unique(),
        sorted(table[SOURCE_COLUMN].unique().to_list()),
    )
    return table


def build_timeseries_table(
    request: schema.IndustrialLoadRequestArgs,
    bronze_root: str | Path,
    as_of: dt.datetime | None = None,
) -> pl.DataFrame:
    """Read every requested source's timeseries bronze and stack them."""
    bronze_root = _resolve_root_uri(bronze_root)
    by_source = {
        params.source: _read_bronze(
            dsgrid_bronze.PROFILES[params.source].timeseries_schema,
            dsgrid_bronze.PROFILES[params.source].timeseries_dataset_name,
            params,
            bronze_root,
            f"dsgrid {params.source} timeseries",
            as_of,
        )
        for params in dsgrid_requests(request.state, request.dsgrid)
    }
    table = union_sources(
        by_source, DsgridIndustrialTimeseriesSilverSchema, TIMESERIES_KEY_COLUMNS
    )
    logger.info(
        "%s: %d row(s), %d NAICS code(s), %d hour(s) from %s",
        TIMESERIES_DATASET_NAME,
        table.height,
        table[SECTOR_COLUMN].n_unique(),
        table["timestamp"].n_unique(),
        sorted(table[SOURCE_COLUMN].unique().to_list()),
    )
    return table


# --------------------------------------------------------------------------- #
# Write + orchestration
# --------------------------------------------------------------------------- #


def write_metadata_silver(
    table: pl.DataFrame,
    params: schema.IndustrialLoadRequestArgs,
    silver_root: str | Path,
    writer: str = DEFAULT_WRITER,
    write_time: dt.datetime | None = None,
) -> ManifestRow:
    """Write the stacked metadata table, keyed by state + the sources it spans."""
    return columnar.write_dataset(
        table,
        DsgridIndustrialMetadataSilverSchema,
        METADATA_DATASET_NAME,
        params,
        _resolve_root_uri(silver_root),
        writer=writer,
        write_time=write_time,
    )


def write_timeseries_silver(
    table: pl.DataFrame,
    params: schema.IndustrialLoadRequestArgs,
    silver_root: str | Path,
    writer: str = DEFAULT_WRITER,
    write_time: dt.datetime | None = None,
) -> ManifestRow:
    """Write the stacked timeseries table, keyed the same way."""
    return columnar.write_dataset(
        table,
        DsgridIndustrialTimeseriesSilverSchema,
        TIMESERIES_DATASET_NAME,
        params,
        _resolve_root_uri(silver_root),
        writer=writer,
        write_time=write_time,
    )


def ingest_industrial_silver(
    request: schema.IndustrialLoadRequestArgs,
    bronze_root: str | Path = DEFAULT_BRONZE_DIR,
    silver_root: str | Path = DEFAULT_SILVER_DIR,
    writer: str = DEFAULT_WRITER,
    write_time: dt.datetime | None = None,
    as_of: dt.datetime | None = None,
) -> list[ManifestRow]:
    """
    Full silver step: stack the requested sources' metadata into one table and
    their timeseries into another, then write both.

    Returns the manifest rows in write order: (metadata, timeseries).
    """
    bronze_root = _resolve_root_uri(bronze_root)
    silver_root = _resolve_root_uri(silver_root)

    rows = [
        write_metadata_silver(
            build_metadata_table(request, bronze_root, as_of),
            request,
            silver_root=silver_root,
            writer=writer,
            write_time=write_time,
        ),
        write_timeseries_silver(
            build_timeseries_table(request, bronze_root, as_of),
            request,
            silver_root=silver_root,
            writer=writer,
            write_time=write_time,
        ),
    ]
    for row in rows:
        logger.info("silver: wrote %s -> %s", row.dataset_name, row.data_uri)
    return rows


def read_metadata_silver(
    request: schema.IndustrialLoadRequestArgs,
    silver_root: str | Path = DEFAULT_SILVER_DIR,
    as_of: dt.datetime | None = None,
) -> pl.DataFrame:
    """Read the previously written stacked metadata table."""
    return columnar.read_dataset(
        DsgridIndustrialMetadataSilverSchema,
        METADATA_DATASET_NAME,
        request,
        _resolve_root_uri(silver_root),
        as_of=as_of,
    )


def read_timeseries_silver(
    request: schema.IndustrialLoadRequestArgs,
    silver_root: str | Path = DEFAULT_SILVER_DIR,
    as_of: dt.datetime | None = None,
) -> pl.DataFrame:
    """Read the previously written stacked timeseries table."""
    return columnar.read_dataset(
        DsgridIndustrialTimeseriesSilverSchema,
        TIMESERIES_DATASET_NAME,
        request,
        _resolve_root_uri(silver_root),
        as_of=as_of,
    )


if __name__ == "__main__":
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    _demo = schema.IndustrialLoadRequestArgs()  # Washington, DC
    for _row in ingest_industrial_silver(_demo):
        print(_row.data_uri)
