"""
Tests for the domain half of the load data-fetch helper.

Covers consolidating the geographies, the slice inventory a request resolves to,
the coverage pre-check, and the copy between stores. Deciding whether to run the
pipeline lives in ``batch_jobs.data_fetch.load`` and is tested with the flow-side
code, since it needs Prefect.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from external_data.data_fetch import geographies, load
from external_data.load_pipeline import schema, silver
from external_data.load_pipeline.comstock import bronze as comstock_bronze
from external_data.load_pipeline.resstock import bronze as resstock_bronze

# Two PUMAs of one state plus one of another, so the state collapse is visible.
_CA_A = "G06001101"
_CA_B = "G06001102"
_NY = "G36000101"


# --------------------------------------------------------------------------- #
# Consolidation
# --------------------------------------------------------------------------- #


def test_consolidate_derives_the_distinct_states() -> None:
    grid = geographies.consolidate([_CA_A, _CA_B, _NY])
    assert [p.puma_gisjoin for p in grid.pumas] == [_CA_A, _CA_B, _NY]
    # dsgrid publishes per state, so two CA PUMAs are one state's work.
    assert grid.states == ("CA", "NY")
    assert grid.state_by_puma[_CA_B] == "CA"


def test_consolidate_drops_a_repeated_puma() -> None:
    # LoadGeographies rejects a repeated PUMA outright, so de-duplicating has to
    # happen before that model is built -- which is this function's job.
    grid = geographies.consolidate([_CA_A, _CA_A, _NY])
    assert [p.puma_gisjoin for p in grid.pumas] == [_CA_A, _NY]
    assert len(grid.geographies.pumas) == 2


def test_consolidate_rejects_a_malformed_gisjoin() -> None:
    with pytest.raises(Exception, match="puma_gisjoin"):
        geographies.consolidate(["not-a-gisjoin"])


# --------------------------------------------------------------------------- #
# The slice inventory
# --------------------------------------------------------------------------- #


def test_plan_slices_covers_the_six_datasets_a_caller_asked_for() -> None:
    """
    The dsgrid bronze is deliberately absent: it is an input to the silver join,
    not something a caller wants back, and a state whose silver exists needs no
    dsgrid read at all. The ingest still writes it -- the flow fetches all three
    sources -- it is simply not planned, fetched or returned here.
    """
    grid = geographies.consolidate([_CA_A, _CA_B, _NY])
    slices = load.plan_slices(grid)
    counts: dict[str, int] = {}
    for entry in slices:
        counts[entry.dataset_name] = counts.get(entry.dataset_name, 0) + 1

    # Building stock: one slice per PUMA.
    for dataset in (
        resstock_bronze.METADATA_DATASET_NAME,
        resstock_bronze.TIMESERIES_DATASET_NAME,
        comstock_bronze.PUMA_METADATA_DATASET_NAME,
        comstock_bronze.TIMESERIES_DATASET_NAME,
    ):
        assert counts[dataset] == 3, dataset
    # Industrial silver: one per state, so the two CA PUMAs share.
    for dataset in (silver.METADATA_DATASET_NAME, silver.TIMESERIES_DATASET_NAME):
        assert counts[dataset] == 2, dataset
    # And nothing else -- no dsgrid bronze.
    assert set(counts) == {
        resstock_bronze.METADATA_DATASET_NAME,
        resstock_bronze.TIMESERIES_DATASET_NAME,
        comstock_bronze.PUMA_METADATA_DATASET_NAME,
        comstock_bronze.TIMESERIES_DATASET_NAME,
        silver.METADATA_DATASET_NAME,
        silver.TIMESERIES_DATASET_NAME,
    }


def test_plan_slices_keys_match_the_pipelines_own_derivation() -> None:
    grid = geographies.consolidate([_CA_A])
    slices = {s.dataset_name: s for s in load.plan_slices(grid)}
    geography = grid.pumas[0]

    # Keyed exactly as the pipeline writes it: a key derived independently would
    # drift and silently miss its own data.
    assert slices[
        resstock_bronze.TIMESERIES_DATASET_NAME
    ].params == silver.resstock_timeseries_request(geography)
    assert slices[
        comstock_bronze.PUMA_METADATA_DATASET_NAME
    ].params == silver.comstock_puma_metadata_request(geography)
    assert slices[silver.METADATA_DATASET_NAME].params == (
        schema.IndustrialLoadRequestArgs(state="CA")
    )


# --------------------------------------------------------------------------- #
# Coverage
# --------------------------------------------------------------------------- #


def test_plan_coverage_on_an_empty_store_has_nothing(tmp_path: Path) -> None:
    grid = geographies.consolidate([_CA_A])
    slices = load.plan_slices(grid)
    cover = load.plan_coverage(slices, str(tmp_path / "empty"))

    assert not cover.complete
    assert cover.covered == ()
    assert len(cover.missing) == len(slices)
