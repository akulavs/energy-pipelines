"""Tests for climate_pipeline catalog registration."""

from __future__ import annotations

from common.storage import catalog

from external_data.climate_pipeline import catalog as cp_catalog
from external_data.climate_pipeline import schema, silver
from external_data.climate_pipeline.era5 import bronze as era5_bronze
from external_data.climate_pipeline.nsrdb import bronze as nsrdb_bronze


def test_register_datasets() -> None:
    catalog.reset()
    cp_catalog.register_datasets()

    # all three dataset types register under their names
    assert {
        era5_bronze.DATASET_NAME,
        nsrdb_bronze.DATASET_NAME,
        silver.DATASET_NAME,
    } <= set(catalog.names())

    # Every write is per point, so every key is a point slice -- ERA5's adds the
    # variables it holds (a fetch may ask for a subset) and NSRDB's the interval
    # (the subtle one -- 30/60-min slices must not collide).
    assert (
        catalog.get(era5_bronze.DATASET_NAME).params_model is schema.Era5PointSliceKey
    )
    assert (
        catalog.get(nsrdb_bronze.DATASET_NAME).params_model is schema.NsrdbPointSliceKey
    )
    assert catalog.get(silver.DATASET_NAME).params_model is schema.PointSliceKey

    # schemas are wired to the right dataset
    assert (
        catalog.get_as(era5_bronze.DATASET_NAME, catalog.DatasetType).schema
        is era5_bronze.Era5LandBronzeSchema
    )
    assert (
        catalog.get_as(nsrdb_bronze.DATASET_NAME, catalog.DatasetType).schema
        is nsrdb_bronze.NsrdbBronzeSchema
    )
    assert (
        catalog.get_as(silver.DATASET_NAME, catalog.DatasetType).schema
        is silver.Era5NsrdbSilverSchema
    )
