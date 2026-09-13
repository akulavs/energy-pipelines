"""
Tests for batch_jobs.data_fetch.climate -- the orchestration half of the helper.

The question here is only *whether the pipeline runs*: the consolidation, the
coverage pre-check and the read-back are covered by
packages/external-data/tests/data_fetch, and the fan-out itself by
tests/batch_jobs/climate_pipeline.

The CDS / NSRDB fetch seams are monkeypatched so everything runs offline, using
the same private seams and committed goldens as the flow tests.
"""

from __future__ import annotations

import datetime
import pathlib

import pandas as pd
import polars as pl
import pytest

from batch_jobs.data_fetch import climate as data_fetch_flow
from common.exceptions import PipelineValueError
from external_data.climate_pipeline import schema, silver
from external_data.climate_pipeline.era5 import bronze as era5_bronze
from external_data.climate_pipeline.nsrdb import bronze as nsrdb_bronze

# Committed fixtures from the package tests, as the flow tests use.
_CP = (
    pathlib.Path(__file__).parents[3] / "packages/external-data/tests/climate_pipeline"
)
_ERA5_BRONZE_GOLDEN = _CP / "era5/fixtures/era5_land_ts_bronze.parquet"
_NSRDB_BRONZE_GOLDEN = _CP / "nsrdb/fixtures/nsrdb_goes_aggregated_bronze.parquet"

_POINTS = ((37.77, -122.42), (37.83, -122.44))
# Wide enough to contain both goldens' own timestamps, so the bronze a run writes
# actually holds rows inside the range its manifest key claims. The ERA5 fake
# replays the golden's day (2025-06-01) verbatim; the NSRDB fake re-stamps the
# golden's January days into each requested year. A range covering neither would
# write files that contradict their own keys.
_START = datetime.date(2023, 1, 1)
_END = datetime.date(2025, 6, 30)


@pytest.fixture
def root_uri(tmp_path: pathlib.Path) -> str:
    return str(tmp_path / "climate_pipeline")


def _fake_era5_fetch(golden: pd.DataFrame):
    def _inner(latitude, longitude, *args, **kwargs) -> pd.DataFrame:
        table = golden.copy()
        table["latitude"] = round(latitude, 1)
        table["longitude"] = round(longitude, 1)
        return table

    return _inner


def _fake_nsrdb_fetch(golden: pl.DataFrame):
    def _inner(latitude, longitude, year, *args, **kwargs) -> pl.DataFrame:
        return golden.with_columns(
            pl.lit(round(latitude, 4), dtype=pl.Float64).alias("latitude"),
            pl.lit(round(longitude, 4), dtype=pl.Float64).alias("longitude"),
            pl.datetime(
                year,
                pl.col("valid_time").dt.month(),
                pl.col("valid_time").dt.day(),
                pl.col("valid_time").dt.hour(),
                pl.col("valid_time").dt.minute(),
            )
            .dt.replace_time_zone("UTC")
            .alias("valid_time"),
        )

    return _inner


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        era5_bronze,
        "_fetch_point_table",
        _fake_era5_fetch(pd.read_parquet(_ERA5_BRONZE_GOLDEN)),
    )
    monkeypatch.setattr(
        nsrdb_bronze,
        "_fetch_point_year_table",
        _fake_nsrdb_fetch(pl.read_parquet(_NSRDB_BRONZE_GOLDEN)),
    )


def _ingest(root_uri: str, points=_POINTS, **kwargs):
    return data_fetch_flow.fetch_and_ingest(
        points, start_date=_START, end_date=_END, root_uri=root_uri, **kwargs
    )


def _fetch(root_uri: str, points=_POINTS, **kwargs):
    return data_fetch_flow.fetch(
        points, start_date=_START, end_date=_END, root_uri=root_uri, **kwargs
    )


def test_an_empty_store_runs_the_pipeline_and_returns_every_dataset(
    root_uri: str, offline: None
) -> None:
    result = _ingest(root_uri)

    assert result.ran_pipeline
    # Silver only: the bronze it is joined from is an input, still ingested but
    # not returned.
    assert result.silver.height > 0
    # Both coordinates sit in one node, so every source is called once -- NSRDB
    # included, since it is snapped to the same grid.
    assert result.points.nodes == (era5_bronze.node(_POINTS[0]),)
    assert result.points.nsrdb_points == result.points.nodes


def test_a_covered_request_skips_the_pipeline_entirely(
    root_uri: str, offline: None, mocker
) -> None:
    first = _ingest(root_uri)
    assert first.ran_pipeline

    # Patched only for the second call, so a run would be loud rather than slow.
    flow = mocker.patch.object(
        data_fetch_flow, "run_climate_pipeline", side_effect=AssertionError
    )
    second = _ingest(root_uri)

    flow.assert_not_called()
    assert not second.ran_pipeline
    assert second.coverage.complete
    # Skipping the run must not change the answer.
    assert second.silver.equals(first.silver)


def test_a_missing_point_runs_the_pipeline_again(
    root_uri: str, offline: None, mocker
) -> None:
    _ingest(root_uri)
    spy = mocker.spy(data_fetch_flow, "run_climate_pipeline")

    # A third coordinate, in a node nothing has fetched.
    result = _ingest(root_uri, points=(*_POINTS, (38.15, -122.25)))

    assert spy.call_count == 1
    assert result.ran_pipeline
    assert not result.coverage.complete
    assert len(result.points.era5_points) == 2


def test_force_refresh_runs_even_when_everything_is_covered(
    root_uri: str, offline: None, mocker
) -> None:
    _ingest(root_uri)
    spy = mocker.spy(data_fetch_flow, "run_climate_pipeline")

    result = _ingest(root_uri, force_refresh=True)

    # The pre-check says there is nothing to do; force_refresh overrides it.
    assert result.coverage.complete
    assert result.ran_pipeline
    assert spy.call_count == 1


def test_a_narrower_stored_range_is_not_treated_as_coverage(
    root_uri: str, offline: None, mocker
) -> None:
    _ingest(root_uri)
    spy = mocker.spy(data_fetch_flow, "run_climate_pipeline")

    # Same points, a range extending past what was stored: present is not
    # covering, so the ERA5 writes no longer answer for it and the run repeats.
    data_fetch_flow.fetch_and_ingest(
        _POINTS,
        start_date=_START,
        end_date=datetime.date(2025, 12, 31),
        root_uri=root_uri,
    )

    assert spy.call_count == 1


def test_rejects_an_interval_the_silver_join_cannot_read(root_uri: str) -> None:
    # Checked before the pre-check, so it fails the same way whether or not the
    # store already has coverage.
    with pytest.raises(PipelineValueError, match="60-minute"):
        _ingest(root_uri, nsrdb=schema.NsrdbRequestArgs(interval=30))


# --------------------------------------------------------------------------- #
# fetch: the read-only door
# --------------------------------------------------------------------------- #


def test_fetch_refuses_when_the_store_has_nothing(
    root_uri: str, offline: None, mocker
) -> None:
    """
    An empty store has no partial answer to give -- and fetch must not quietly
    start an ingest, which is the whole reason it is a separate call.
    """
    flow = mocker.patch.object(
        data_fetch_flow, "run_climate_pipeline", side_effect=AssertionError
    )

    with pytest.raises(PipelineValueError, match="has nothing for this request"):
        _fetch(root_uri)

    flow.assert_not_called()


def test_fetch_returns_the_datasets_once_they_are_there(
    root_uri: str, offline: None, mocker
) -> None:
    _ingest(root_uri)
    flow = mocker.patch.object(
        data_fetch_flow, "run_climate_pipeline", side_effect=AssertionError
    )

    result = _fetch(root_uri)

    flow.assert_not_called()
    # Never ingests, so this is always False from fetch.
    assert not result.ran_pipeline
    assert result.silver.height > 0


def test_fetch_returns_what_is_there_and_reports_what_is_not(
    root_uri: str, offline: None
) -> None:
    _ingest(root_uri)
    absent = (38.15, -122.25)

    # Two coordinates present, one in a node nothing has ingested.
    result = _fetch(root_uri, points=(*_POINTS, absent))

    # The locations the store has still come back...
    assert result.silver.height > 0
    # ...and the one it lacks is named on the coverage it returns.
    assert era5_bronze.node(absent) in result.coverage.silver.missing
    assert era5_bronze.node(_POINTS[0]) in result.coverage.silver.covered
    # Named in the message an engineer sees, not only in the structured field.
    # Asserted on the rendering rather than on captured log text: Prefect
    # reconfigures logger propagation, so whether a record reaches caplog's root
    # handler depends on what ran before it.
    #
    # The message names the *node*, since that is the grain the datasets are
    # stored at -- which is why the consolidation reports the mapping from the
    # coordinate that was typed to the node it resolved to.
    node = era5_bronze.node(absent)
    rendered = data_fetch_flow._describe_missing(result.coverage)
    assert f"{node[0]:g}" in rendered and f"{node[1]:g}" in rendered


def test_fetch_refuses_a_range_wider_than_anything_stored(
    root_uri: str, offline: None
) -> None:
    _ingest(root_uri)

    # Present is not covering: the stored writes fall short of this range, and
    # every location falls short, so there is nothing partial to return.
    with pytest.raises(PipelineValueError, match="has nothing for this request"):
        data_fetch_flow.fetch(
            _POINTS,
            start_date=_START,
            end_date=datetime.date(2025, 12, 31),
            root_uri=root_uri,
        )


def test_ingesting_a_new_point_leaves_the_existing_silver_alone(
    root_uri: str, offline: None
) -> None:
    """
    The flow's silver join rebuilds every node in the request it is handed -- it
    has no per-node coverage check -- so a request carrying already-covered nodes
    would rebuild their silver, superseding good tables and orphaning the parquet
    behind them. Only the nodes needing work are passed.

    Counted on the manifest sidecars rather than through ``query_manifest``,
    which returns the latest row per key and so hides a rebuild entirely.
    """
    manifests = pathlib.Path(root_uri) / "_manifests"

    def written(dataset_name: str) -> int:
        return len(list((manifests / dataset_name).glob("*.json")))

    _ingest(root_uri, points=(_POINTS[0],))
    assert written(silver.DATASET_NAME) == 1

    # A second node, in a node nothing has ingested.
    result = _ingest(root_uri, points=(_POINTS[0], (38.15, -122.25)))

    assert result.ran_pipeline
    # One write per node, not a rebuild of the first.
    assert written(silver.DATASET_NAME) == 2
    assert written(era5_bronze.DATASET_NAME) == 2
    # And both nodes still come back.
    assert result.silver.select("era5_latitude", "era5_longitude").unique().height == 2


def test_force_refresh_still_rebuilds_every_node(root_uri: str, offline: None) -> None:
    manifests = pathlib.Path(root_uri) / "_manifests"
    _ingest(root_uri, points=(_POINTS[0],))

    _ingest(root_uri, points=(_POINTS[0],), force_refresh=True)

    # force_refresh exists to redo work, so the second write is the point.
    assert len(list((manifests / silver.DATASET_NAME).glob("*.json"))) == 2


def test_a_past_as_of_does_not_hide_what_the_ingest_just_wrote(
    root_uri: str, offline: None
) -> None:
    """
    The pipeline stamps ``write_time = now``, and a read resolves
    ``write_time <= as_of`` -- so threading a past ``as_of`` into the post-ingest
    read filters out the very rows the fetch just paid for.
    ``run_climate_pipeline`` guards its own silver step the same way.
    """
    past = datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC)

    result = data_fetch_flow.fetch_and_ingest(
        _POINTS, start_date=_START, end_date=_END, root_uri=root_uri, as_of=past
    )

    assert result.ran_pipeline
    # Without the guard this raises: nothing was written at or before 2020.
    assert result.silver.height > 0


def test_a_past_as_of_is_honoured_when_nothing_is_ingested(
    root_uri: str, offline: None
) -> None:
    """
    The bound is only ignored for a run's own output. With nothing ingested there
    is nothing new to see, so time travel works as asked -- a past ``as_of``
    still excludes writes made after it.
    """
    _ingest(root_uri)
    past = datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC)

    with pytest.raises(PipelineValueError):
        data_fetch_flow.fetch(
            _POINTS, start_date=_START, end_date=_END, root_uri=root_uri, as_of=past
        )
