"""
Tests for the load pipeline's catalog registration.

One ``register_datasets()`` wires up the whole pipeline -- the three sectors'
bronze datasets plus the silver join -- so this is the single place that asserts
the registry contents. The per-source bronze tests no longer each register their
own.
"""

from __future__ import annotations

import collections.abc

import pytest

from common.storage import catalog
from external_data.load_pipeline import catalog as load_catalog
from external_data.load_pipeline import schema, silver
from external_data.load_pipeline.comstock import bronze as comstock_bronze
from external_data.load_pipeline.dsgrid import bronze as dsgrid_bronze
from external_data.load_pipeline.resstock import bronze as resstock_bronze


@pytest.fixture(autouse=True)
def _clean_catalog() -> collections.abc.Generator[None]:
    """Each test starts and ends with an empty catalog."""
    catalog.reset()
    yield
    catalog.reset()


def _dsgrid_names() -> list[str]:
    return [
        name
        for profile in dsgrid_bronze.PROFILES.values()
        for name in (profile.metadata_dataset_name, profile.timeseries_dataset_name)
    ]


def test_registers_every_dataset_the_pipeline_touches() -> None:
    load_catalog.register_datasets()

    expected = {
        resstock_bronze.METADATA_DATASET_NAME,
        resstock_bronze.TIMESERIES_DATASET_NAME,
        comstock_bronze.PUMA_METADATA_DATASET_NAME,
        comstock_bronze.TIMESERIES_DATASET_NAME,
        *_dsgrid_names(),
        silver.METADATA_DATASET_NAME,
        silver.TIMESERIES_DATASET_NAME,
    }
    # Equality, not a subset: a dataset registered without being named here is as
    # much a gap as one named without being registered.
    assert expected == set(catalog.names())
    # 2 ResStock + 2 ComStock + 4 dsgrid bronze + 2 dsgrid silver.
    assert len(expected) == 10


def test_resstock_types() -> None:
    load_catalog.register_datasets()
    assert (
        catalog.get_as(
            resstock_bronze.METADATA_DATASET_NAME, catalog.DatasetType
        ).schema
        is resstock_bronze.ResstockMetadataBronzeSchema
    )
    assert (
        catalog.get(resstock_bronze.TIMESERIES_DATASET_NAME).params_model
        is resstock_bronze.ResstockPumaTimeseriesRequestArgs
    )


def test_comstock_types() -> None:
    load_catalog.register_datasets()
    assert (
        catalog.get_as(
            comstock_bronze.PUMA_METADATA_DATASET_NAME, catalog.DatasetType
        ).schema
        is comstock_bronze.ComstockPumaMetadataBronzeSchema
    )
    assert (
        catalog.get(comstock_bronze.TIMESERIES_DATASET_NAME).params_model
        is comstock_bronze.ComstockPumaTimeseriesRequestArgs
    )


def test_dsgrid_types() -> None:
    load_catalog.register_datasets()
    for profile in dsgrid_bronze.PROFILES.values():
        for name, dataset_schema in (
            (profile.metadata_dataset_name, profile.metadata_schema),
            (profile.timeseries_dataset_name, profile.timeseries_schema),
        ):
            entry = catalog.get_as(name, catalog.DatasetType)
            assert entry.schema is dataset_schema
            assert entry.params_model is dsgrid_bronze.DsgridRequestArgs


def test_silver_types() -> None:
    """One silver table per kind, each keyed by state + the sources it stacks."""
    load_catalog.register_datasets()
    for name, sch in (
        (silver.METADATA_DATASET_NAME, silver.DsgridIndustrialMetadataSilverSchema),
        (silver.TIMESERIES_DATASET_NAME, silver.DsgridIndustrialTimeseriesSilverSchema),
    ):
        entry = catalog.get_as(name, catalog.DatasetType)
        assert entry.schema is sch
        assert entry.params_model is schema.IndustrialLoadRequestArgs


def test_every_registration_carries_a_description() -> None:
    load_catalog.register_datasets()
    missing = [t.name for t in catalog.all_types() if not t.description.strip()]
    assert not missing


def test_registering_twice_fails_loud() -> None:
    """The duplicate guard is what catches a composition root wired in twice."""
    load_catalog.register_datasets()
    with pytest.raises(ValueError, match="already registered"):
        load_catalog.register_datasets()
