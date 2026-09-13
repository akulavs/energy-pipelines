"""
Catalog registration for the climate pipeline dataset types.

ERA5 bronze is keyed by
:class:`~external_data.climate_pipeline.schema.Era5PointSliceKey` (a point slice
plus the variables that write holds, since a fetch may ask for a subset of the
curated set); the ERA5 x NSRDB silver join is keyed by the plain point slice.
NSRDB bronze is keyed by
:class:`~external_data.climate_pipeline.schema.NsrdbDatasetKey` (that request plus
the fetch ``interval``), so 30- and 60-minute fetches are distinct datasets.
"""

from __future__ import annotations

from common.storage import catalog

from external_data.climate_pipeline import schema, silver
from external_data.climate_pipeline.era5 import bronze as era5_bronze
from external_data.climate_pipeline.nsrdb import bronze as nsrdb_bronze


def register_datasets() -> None:
    """
    Register the climate pipeline's bronze and silver dataset types into the
    shared catalog.
    """
    catalog.register(
        catalog.DatasetType(
            name=era5_bronze.DATASET_NAME,
            schema=era5_bronze.Era5LandBronzeSchema,
            params_model=schema.Era5PointSliceKey,
            description=(
                "ERA5-Land hourly point timeseries (bronze): one row per "
                "timestamp per grid node, partitioned by location."
            ),
        )
    )
    catalog.register(
        catalog.DatasetType(
            name=nsrdb_bronze.DATASET_NAME,
            schema=nsrdb_bronze.NsrdbBronzeSchema,
            params_model=schema.NsrdbPointSliceKey,
            description=(
                "NSRDB GOES Aggregated solar timeseries (bronze): one row per "
                "timestamp per 4 km cell, partitioned by location. Keyed by "
                "request + interval, so 30- and 60-minute fetches are distinct."
            ),
        )
    )
    catalog.register(
        catalog.DatasetType(
            name=silver.DATASET_NAME,
            schema=silver.Era5NsrdbSilverSchema,
            params_model=schema.PointSliceKey,
            description=(
                "ERA5-Land x NSRDB silver: one row per ERA5 0.1 deg grid node x "
                "hour joining ERA5 weather (unit-harmonized) with NSRDB "
                "solar/irradiance (snapped onto the ERA5 grid; None where "
                "uncovered). Written one entry per grid node, so the table is "
                "built (and read) node by node rather than all at once."
            ),
        )
    )
