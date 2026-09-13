"""
Catalog registration for the load pipeline dataset types.

One place registers the whole pipeline: the three sectors' bronze datasets
(ResStock residential, ComStock commercial, dsgrid industrial -- both of its source
files) and the two silver tables that stack the dsgrid pair by kind.
"""

from __future__ import annotations

from common.storage import catalog

from external_data.load_pipeline import schema, silver
from external_data.load_pipeline.comstock import bronze as comstock_bronze
from external_data.load_pipeline.dsgrid import bronze as dsgrid_bronze
from external_data.load_pipeline.resstock import bronze as resstock_bronze

# Catalog copy for the dsgrid sources, keyed by source rather than held on the
# profile -- this is its only reader.
_DSGRID_METADATA_DESCRIPTIONS: dict[dsgrid_bronze.DsgridSource, str] = {
    dsgrid_bronze.DsgridSource.INDUSTRIAL: (
        "dsgrid 2018 EFS industrial metadata (bronze): one row per county "
        "x NAICS subsector with annual electricity totals by end use, "
        "as one unpartitioned file per state."
    ),
    dsgrid_bronze.DsgridSource.GAPS: (
        "dsgrid 2018 EFS industrial-gaps metadata (bronze): one row per "
        "county x non-manufacturing NAICS sector with the annual "
        "electricity total, as one unpartitioned file per state."
    ),
}
_DSGRID_TIMESERIES_DESCRIPTIONS: dict[dsgrid_bronze.DsgridSource, str] = {
    dsgrid_bronze.DsgridSource.INDUSTRIAL: (
        "dsgrid 2018 EFS industrial timeseries (bronze): one row per county "
        "x NAICS subsector x hour with the 12 industrial electricity end "
        "uses, as one unpartitioned file per state."
    ),
    dsgrid_bronze.DsgridSource.GAPS: (
        "dsgrid 2018 EFS industrial-gaps timeseries (bronze): one row per "
        "county x non-manufacturing NAICS sector x hour of electricity "
        "demand, as one unpartitioned file per state."
    ),
}


def _register_resstock() -> None:
    catalog.register(
        catalog.DatasetType(
            name=resstock_bronze.METADATA_DATASET_NAME,
            schema=resstock_bronze.ResstockMetadataBronzeSchema,
            params_model=resstock_bronze.ResstockMetadataRequestArgs,
            description=(
                "ResStock per-building metadata + annual results (bronze): one "
                "row per building model of one PUMA, keyed by that PUMA. Cut "
                "from the state file OEDI publishes, which one ingest reads "
                "once for every PUMA of that state a run asked for."
            ),
        )
    )
    catalog.register(
        catalog.DatasetType(
            name=resstock_bronze.TIMESERIES_DATASET_NAME,
            schema=resstock_bronze.ResstockTimeseriesBronzeSchema,
            params_model=resstock_bronze.ResstockPumaTimeseriesRequestArgs,
            description=(
                "ResStock PUMA-batched timeseries (bronze): every building in a "
                "PUMA x 15-minute timestep, as one write keyed by that PUMA."
            ),
        )
    )


def _register_comstock() -> None:
    catalog.register(
        catalog.DatasetType(
            name=comstock_bronze.PUMA_METADATA_DATASET_NAME,
            schema=comstock_bronze.ComstockPumaMetadataBronzeSchema,
            params_model=comstock_bronze.ComstockPumaMetadataRequestArgs,
            description=(
                "ComStock per-PUMA metadata + annual results (bronze): one row "
                "per building model, census-tract duplication collapsed and "
                "weight summed within the PUMA, keyed by that PUMA. The table "
                "to join to comstock_timeseries_bronze on bldg_id."
            ),
        )
    )
    catalog.register(
        catalog.DatasetType(
            name=comstock_bronze.TIMESERIES_DATASET_NAME,
            schema=comstock_bronze.ComstockTimeseriesBronzeSchema,
            params_model=comstock_bronze.ComstockPumaTimeseriesRequestArgs,
            description=(
                "ComStock PUMA-batched timeseries (bronze): every building in a "
                "PUMA x 15-minute timestep, as one write keyed by that PUMA."
            ),
        )
    )


def _register_dsgrid() -> None:
    for source, profile in dsgrid_bronze.PROFILES.items():
        catalog.register(
            catalog.DatasetType(
                name=profile.metadata_dataset_name,
                schema=profile.metadata_schema,
                params_model=dsgrid_bronze.DsgridRequestArgs,
                description=_DSGRID_METADATA_DESCRIPTIONS[source],
            )
        )
        catalog.register(
            catalog.DatasetType(
                name=profile.timeseries_dataset_name,
                schema=profile.timeseries_schema,
                params_model=dsgrid_bronze.DsgridRequestArgs,
                description=_DSGRID_TIMESERIES_DESCRIPTIONS[source],
            )
        )


def _register_silver() -> None:
    catalog.register(
        catalog.DatasetType(
            name=silver.METADATA_DATASET_NAME,
            schema=silver.DsgridIndustrialMetadataSilverSchema,
            params_model=schema.IndustrialLoadRequestArgs,
            description=(
                "dsgrid industrial metadata silver: one row per county x NAICS "
                "code, stacking the manufacturing and non-manufacturing files "
                "under a harmonised naics_code with a source discriminator. The "
                "12 end-use columns are null on non-manufacturing rows. "
                "Unpartitioned: one file per state."
            ),
        )
    )
    catalog.register(
        catalog.DatasetType(
            name=silver.TIMESERIES_DATASET_NAME,
            schema=silver.DsgridIndustrialTimeseriesSilverSchema,
            params_model=schema.IndustrialLoadRequestArgs,
            description=(
                "dsgrid industrial timeseries silver: one row per county x NAICS "
                "code x hour, stacking both source files the same way. "
                "electricity_total_mwh is populated throughout, so summing it "
                "over a county x hour gives the whole industrial load. Keeps "
                "dsgrid's native hourly 2012 national-clock axis. Unpartitioned."
            ),
        )
    )


def register_datasets() -> None:
    """
    Register the load pipeline's bronze and silver dataset types into the shared
    catalog.
    """
    _register_resstock()
    _register_comstock()
    _register_dsgrid()
    _register_silver()
