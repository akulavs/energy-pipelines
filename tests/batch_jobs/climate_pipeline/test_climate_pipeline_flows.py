"""
Integration tests for batch_jobs.climate_pipeline.flows.

The CDS / NSRDB fetch seams are monkeypatched so the flows run offline. Exercises
the full write + flow-manifest lifecycle for all four flows; the fetch/parse and
join correctness are covered by packages/external-data/tests/climate_pipeline.
"""

from __future__ import annotations

import contextlib
import datetime
import pathlib
import threading

import pandas as pd
import polars as pl
import pytest

from batch_jobs.climate_pipeline import flows
from batch_jobs.climate_pipeline.flows import (
    era5_nsrdb_silver,
    ingest_era5,
    ingest_nsrdb,
    run_climate_pipeline,
)
from common.storage.flow_manifest import FlowStatus, query_flow_manifests
from common.storage.manifest import query_manifest
from external_data.climate_pipeline import failures, point_manifest, schema, silver
from external_data.climate_pipeline.era5 import bronze as era5_bronze
from external_data.climate_pipeline.nsrdb import bronze as nsrdb_bronze

# Committed fixtures from the package tests.
_CP = (
    pathlib.Path(__file__).parents[3] / "packages/external-data/tests/climate_pipeline"
)
_ERA5_BRONZE_GOLDEN = _CP / "era5/fixtures/era5_land_ts_bronze.parquet"
_NSRDB_BRONZE_GOLDEN = _CP / "nsrdb/fixtures/nsrdb_goes_aggregated_bronze.parquet"
_ERA5_SILVER_SAMPLE = _CP / "silver/fixtures/era5_bronze_sample.parquet"
_NSRDB_SILVER_SAMPLE = _CP / "silver/fixtures/nsrdb_bronze_sample.parquet"
_SILVER_GOLDEN = _CP / "silver/fixtures/era5_nsrdb_silver_golden.parquet"

# The grid the silver samples were built for.
_SILVER_POINTS = ((37.8, -122.4), (37.7, -122.4), (37.9, -122.4))
_SILVER_START = datetime.date(2023, 6, 1)
_SILVER_END = datetime.date(2025, 6, 1)


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


# --------------------------------------------------------------------------- #
# 1. ERA5 bronze flow
# --------------------------------------------------------------------------- #


def test_era5_bronze_flow_writes(
    root_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    golden = pd.read_parquet(_ERA5_BRONZE_GOLDEN)
    monkeypatch.setattr(era5_bronze, "_fetch_point_table", _fake_era5_fetch(golden))

    ingest_era5(
        request=schema.ClimatePipelineRequestArgs(
            points=((37.45, -122.25),),
            start_date=datetime.date(2025, 6, 1),
            end_date=datetime.date(2025, 6, 1),
        ),
        root_uri=root_uri,
    )

    rows = query_manifest(dataset_name=era5_bronze.DATASET_NAME, root_uri=root_uri)
    assert len(rows) == 1
    flow = query_flow_manifests(root_uri=root_uri, status=FlowStatus.COMPLETED)
    assert len(flow) == 1
    assert flow[0].metadata["output_write_ids"] == [rows[0].write_id]


# --------------------------------------------------------------------------- #
# 2. NSRDB bronze flow
# --------------------------------------------------------------------------- #


def test_nsrdb_bronze_flow_writes(
    root_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    golden = pl.read_parquet(_NSRDB_BRONZE_GOLDEN)
    monkeypatch.setattr(
        nsrdb_bronze, "_fetch_point_year_table", _fake_nsrdb_fetch(golden)
    )

    ingest_nsrdb(
        request=schema.ClimatePipelineRequestArgs(
            points=((37.77, -122.42),),
            start_date=datetime.date(2023, 1, 1),
            end_date=datetime.date(2023, 1, 31),
        ),
        root_uri=root_uri,
    )

    rows = query_manifest(dataset_name=nsrdb_bronze.DATASET_NAME, root_uri=root_uri)
    assert len(rows) == 1
    flow = query_flow_manifests(root_uri=root_uri, status=FlowStatus.COMPLETED)
    assert len(flow) == 1


def test_nsrdb_bronze_flow_skips_out_of_coverage(
    root_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A range past NSRDB coverage -> no fetch, no dataset, and the flow completes
    # with a "skipped" note. Derived from the coverage window so a coverage bump
    # can't silently turn this into a different test.
    beyond = schema.NSRDB_MAX_DATE.year + 1
    called: list[object] = []
    monkeypatch.setattr(
        nsrdb_bronze,
        "_fetch_point_year_table",
        lambda *a, **k: called.append(1),
    )

    ingest_nsrdb(
        request=schema.ClimatePipelineRequestArgs(
            points=((37.8, -122.4),),
            start_date=datetime.date(beyond, 1, 1),
            end_date=datetime.date(beyond, 12, 31),
        ),
        root_uri=root_uri,
    )

    assert not called  # nothing fetched
    assert not query_manifest(dataset_name=nsrdb_bronze.DATASET_NAME, root_uri=root_uri)
    flow = query_flow_manifests(root_uri=root_uri, status=FlowStatus.COMPLETED)
    assert len(flow) == 1
    assert "skipped" in flow[0].metadata


# --------------------------------------------------------------------------- #
# 3. Silver flow
# --------------------------------------------------------------------------- #


def _rows_for(frame: pl.DataFrame, point: tuple[float, float]) -> pl.DataFrame:
    """The seed frame's rows for one point, matched at the grid-node grain."""
    node = era5_bronze.node(point)
    return frame.filter(
        (pl.col("latitude").round(era5_bronze.NODE_DECIMALS) == node[0])
        & (pl.col("longitude").round(era5_bronze.NODE_DECIMALS) == node[1])
    )


def _seed_silver_bronze(root_uri: str) -> schema.ClimatePipelineRequestArgs:
    request = schema.ClimatePipelineRequestArgs(
        points=_SILVER_POINTS, start_date=_SILVER_START, end_date=_SILVER_END
    )
    seed_time = datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC)
    # One entry per point, as the bronze flows write it -- a batch key is not
    # readable per point, so seeding one would test a layout we no longer produce.
    era5_frame = pl.read_parquet(_ERA5_SILVER_SAMPLE)
    for point in era5_bronze.unique_points(request):
        era5_bronze.write_point_bronze(
            _rows_for(era5_frame, point).to_pandas(),
            point,
            request,
            root_uri,
            writer="seed",
            write_time=seed_time,
        )
    nreq = silver.nsrdb_request(request)
    assert nreq is not None
    nsrdb_args = schema.NsrdbRequestArgs(interval=schema.NSRDB_SILVER_INTERVAL)
    nsrdb_frame = pl.read_parquet(_NSRDB_SILVER_SAMPLE)
    points = dict.fromkeys(point for point, _year in nsrdb_bronze.request_units(nreq))
    for point in points:
        nsrdb_bronze.write_point_bronze(
            _rows_for(nsrdb_frame, point),
            point,
            nreq,
            nsrdb_args,
            root_uri,
            writer="seed",
            write_time=seed_time,
        )
    return request


def test_silver_flow_round_trips(root_uri: str) -> None:
    request = _seed_silver_bronze(root_uri)

    era5_nsrdb_silver(request=request, root_uri=root_uri)

    back = silver.read_points_silver(request, root_uri)
    golden = pl.read_parquet(_SILVER_GOLDEN)
    assert back.height == golden.height == 21
    # Silver is written one entry per grid node, so the run produced as many
    # writes as the request has distinct nodes.
    assert len(query_manifest(dataset_name=silver.DATASET_NAME, root_uri=root_uri)) == 3

    flow = query_flow_manifests(root_uri=root_uri, status=FlowStatus.COMPLETED)
    assert len(flow) == 1
    # Bronze is one write per point, so lineage names every point it read, not a
    # single batch row per source. Derived from the seed so it cannot drift.
    era5_writes = query_manifest(
        dataset_name=era5_bronze.DATASET_NAME, root_uri=root_uri
    )
    nsrdb_writes = query_manifest(
        dataset_name=nsrdb_bronze.DATASET_NAME, root_uri=root_uri
    )
    assert len(flow[0].input_ids) == len(era5_writes) + len(nsrdb_writes)
    assert set(flow[0].input_ids) == {
        row.write_id for row in (*era5_writes, *nsrdb_writes)
    }


def test_silver_flow_raises_when_bronze_missing(root_uri: str) -> None:
    request = schema.ClimatePipelineRequestArgs(
        points=_SILVER_POINTS, start_date=_SILVER_START, end_date=_SILVER_END
    )
    with pytest.raises(RuntimeError, match="ingest the ERA5-Land bronze"):
        era5_nsrdb_silver(request=request, root_uri=root_uri)
    failed = query_flow_manifests(root_uri=root_uri, status=FlowStatus.FAILED)
    assert len(failed) == 1


# --------------------------------------------------------------------------- #
# 4. Full pipeline
# --------------------------------------------------------------------------- #


def test_pipeline_runs_all_three(
    root_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    run_climate_pipeline(
        request=schema.ClimatePipelineRequestArgs(
            points=((37.77, -122.42),),
            start_date=datetime.date(2023, 1, 1),
            end_date=datetime.date(2023, 1, 31),
        ),
        root_uri=root_uri,
    )

    # all three datasets landed under the single root_uri
    assert query_manifest(dataset_name=era5_bronze.DATASET_NAME, root_uri=root_uri)
    assert query_manifest(dataset_name=nsrdb_bronze.DATASET_NAME, root_uri=root_uri)
    silver_rows = query_manifest(dataset_name=silver.DATASET_NAME, root_uri=root_uri)
    assert len(silver_rows) == 1


def test_pipeline_rejects_non_hourly_nsrdb(root_uri: str) -> None:
    # Silver joins the 60-minute NSRDB, so the pipeline must fail fast on a
    # 30-minute config -- before ingesting anything -- not after ERA5 + NSRDB run.
    with pytest.raises(RuntimeError, match="NSRDB interval=60"):
        run_climate_pipeline(
            request=schema.ClimatePipelineRequestArgs(
                points=((37.77, -122.42),),
                start_date=datetime.date(2023, 1, 1),
                end_date=datetime.date(2023, 1, 31),
            ),
            nsrdb=schema.NsrdbRequestArgs(interval=30),
            root_uri=root_uri,
        )

    # nothing was ingested (validation ran before any subflow)...
    assert not query_manifest(dataset_name=era5_bronze.DATASET_NAME, root_uri=root_uri)
    assert not query_manifest(dataset_name=nsrdb_bronze.DATASET_NAME, root_uri=root_uri)
    # ...and the pipeline recorded a FAILED flow-manifest.
    failed = query_flow_manifests(root_uri=root_uri, status=FlowStatus.FAILED)
    assert len(failed) == 1
    assert failed[0].flow_name == "run_climate_pipeline"


# --------------------------------------------------------------------------- #
# 5. Manifest-driven skip / refresh
# --------------------------------------------------------------------------- #


def _counting_era5_fetch(golden: pd.DataFrame, calls: list) -> object:
    """The ERA5 fetch stub, recording which points it was actually asked for."""
    inner = _fake_era5_fetch(golden)

    def _wrapped(latitude, longitude, *args, **kwargs):
        calls.append((latitude, longitude))
        return inner(latitude, longitude, *args, **kwargs)

    return _wrapped


def _era5_request(
    points: tuple[tuple[float, float], ...],
    end: datetime.date = datetime.date(2025, 6, 1),
) -> schema.ClimatePipelineRequestArgs:
    return schema.ClimatePipelineRequestArgs(
        points=points, start_date=datetime.date(2025, 6, 1), end_date=end
    )


def test_era5_bronze_flow_skips_points_already_covered(
    root_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    golden = pd.read_parquet(_ERA5_BRONZE_GOLDEN)
    calls: list = []
    monkeypatch.setattr(
        era5_bronze, "_fetch_point_table", _counting_era5_fetch(golden, calls)
    )
    points = ((37.42, -122.23), (40.72, -73.96))

    ingest_era5(request=_era5_request(points), root_uri=root_uri)
    assert len(calls) == 2

    # Same request again: the manifest already covers both, so nothing is fetched
    # and no second write appears.
    calls.clear()
    ingest_era5(request=_era5_request(points), root_uri=root_uri)
    assert calls == []
    assert (
        len(query_manifest(dataset_name=era5_bronze.DATASET_NAME, root_uri=root_uri))
        == 2
    )


def test_era5_bronze_flow_fetches_only_the_added_points(
    root_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    golden = pd.read_parquet(_ERA5_BRONZE_GOLDEN)
    calls: list = []
    monkeypatch.setattr(
        era5_bronze, "_fetch_point_table", _counting_era5_fetch(golden, calls)
    )
    ingest_era5(request=_era5_request(((37.42, -122.23),)), root_uri=root_uri)

    calls.clear()
    ingest_era5(
        request=_era5_request(((37.42, -122.23), (40.72, -73.96), (34.03, -118.22))),
        root_uri=root_uri,
    )
    assert set(calls) == {(40.72, -73.96), (34.03, -118.22)}


def test_era5_bronze_flow_force_refresh_refetches_everything(
    root_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    golden = pd.read_parquet(_ERA5_BRONZE_GOLDEN)
    calls: list = []
    monkeypatch.setattr(
        era5_bronze, "_fetch_point_table", _counting_era5_fetch(golden, calls)
    )
    request = _era5_request(((37.42, -122.23),))
    ingest_era5(request=request, root_uri=root_uri)

    calls.clear()
    ingest_era5(request=request, root_uri=root_uri, force_refresh=True)
    assert len(calls) == 1
    # A refresh is a new immutable *version*, not an overwrite: both writes
    # survive, so an as_of reader pinned before the refresh still resolves the
    # old one. The default latest-per-params view still collapses to one.
    versions = query_manifest(
        dataset_name=era5_bronze.DATASET_NAME,
        root_uri=root_uri,
        latest_per_params=False,
    )
    assert len(versions) == 2
    latest = query_manifest(dataset_name=era5_bronze.DATASET_NAME, root_uri=root_uri)
    assert len(latest) == 1


def test_era5_bronze_flow_refetches_when_the_range_widens(
    root_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    golden = pd.read_parquet(_ERA5_BRONZE_GOLDEN)
    calls: list = []
    monkeypatch.setattr(
        era5_bronze, "_fetch_point_table", _counting_era5_fetch(golden, calls)
    )
    points = ((37.42, -122.23),)
    ingest_era5(request=_era5_request(points), root_uri=root_uri)

    calls.clear()
    ingest_era5(
        request=_era5_request(points, end=datetime.date(2025, 6, 30)),
        root_uri=root_uri,
    )
    assert len(calls) == 1


def test_era5_bronze_flow_reuses_when_the_variable_set_narrows(
    root_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    golden = pd.read_parquet(_ERA5_BRONZE_GOLDEN)
    calls: list = []
    monkeypatch.setattr(
        era5_bronze, "_fetch_point_table", _counting_era5_fetch(golden, calls)
    )
    points = ((37.42, -122.23),)
    ingest_era5(request=_era5_request(points), root_uri=root_uri)

    # The stored write holds every curated variable, so it answers a request for
    # two of them: dropping variables costs nothing.
    calls.clear()
    ingest_era5(
        request=_era5_request(points),
        era5=schema.Era5RequestArgs(variables=("2m_temperature", "snow_cover")),
        root_uri=root_uri,
    )
    assert calls == []


def test_era5_bronze_flow_refetches_when_the_variable_set_widens(
    root_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    golden = pd.read_parquet(_ERA5_BRONZE_GOLDEN)
    calls: list = []
    monkeypatch.setattr(
        era5_bronze, "_fetch_point_table", _counting_era5_fetch(golden, calls)
    )
    points = ((37.42, -122.23),)
    ingest_era5(
        request=_era5_request(points),
        era5=schema.Era5RequestArgs(variables=("2m_temperature",)),
        root_uri=root_uri,
    )

    # The stored write holds one variable, so it cannot answer a request for the
    # whole set -- without the recorded selection this point would read as
    # covered and the other twelve variables would stay null forever.
    calls.clear()
    ingest_era5(request=_era5_request(points), root_uri=root_uri)
    assert len(calls) == 1


def test_run_climate_pipeline_exposes_force_refresh() -> None:
    # It has to reach the run form, or a full-pipeline run cannot refresh.
    from prefect.utilities.callables import parameter_schema

    assert "force_refresh" in parameter_schema(run_climate_pipeline.fn).properties


def test_nsrdb_bronze_flow_skips_points_already_covered(
    root_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    golden = pl.read_parquet(_NSRDB_BRONZE_GOLDEN)
    inner = _fake_nsrdb_fetch(golden)
    calls: list = []

    def _counting(latitude, longitude, year, *args, **kwargs):
        calls.append((latitude, longitude, year))
        return inner(latitude, longitude, year, *args, **kwargs)

    monkeypatch.setattr(nsrdb_bronze, "_fetch_point_year_table", _counting)
    request = schema.ClimatePipelineRequestArgs(
        points=((37.77, -122.42),),
        start_date=datetime.date(2023, 1, 1),
        end_date=datetime.date(2023, 1, 31),
    )

    ingest_nsrdb(request=request, root_uri=root_uri)
    assert len(calls) == 1

    calls.clear()
    ingest_nsrdb(request=request, root_uri=root_uri)
    assert calls == []

    # force_refresh overrides the manifest and fetches the point-year again.
    ingest_nsrdb(request=request, root_uri=root_uri, force_refresh=True)
    assert len(calls) == 1


# --------------------------------------------------------------------------- #
# 6. Partial-failure tolerance
# --------------------------------------------------------------------------- #


def _failing_era5_fetch(
    golden: pd.DataFrame, dead: tuple[float, float], attempts: dict
):
    """ERA5 stub where one point fails permanently and the rest succeed."""
    inner = _fake_era5_fetch(golden)

    def _wrapped(latitude, longitude, *args, **kwargs):
        point = (latitude, longitude)
        attempts[point] = attempts.get(point, 0) + 1
        if point == dead:
            raise failures.PermanentFetchError(
                "CDS cannot serve this request: Request has not produced any data."
            )
        return inner(latitude, longitude, *args, **kwargs)

    return _wrapped


def _era5_request(points, end=datetime.date(2025, 6, 1)):
    return schema.ClimatePipelineRequestArgs(
        points=points, start_date=datetime.date(2025, 6, 1), end_date=end
    )


def test_era5_flow_completes_when_one_point_fails_permanently(
    root_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The whole point of #4: one dead coordinate must not cost the other points.
    golden = pd.read_parquet(_ERA5_BRONZE_GOLDEN)
    dead = (0.0, -140.0)  # mid-Pacific: valid coordinates, nothing to serve
    attempts: dict = {}
    monkeypatch.setattr(
        era5_bronze, "_fetch_point_table", _failing_era5_fetch(golden, dead, attempts)
    )

    failed = ingest_era5(
        request=_era5_request(((37.42, -122.23), (40.72, -73.96), dead)),
        root_uri=root_uri,
    )

    rows = query_manifest(dataset_name=era5_bronze.DATASET_NAME, root_uri=root_uri)
    assert len(rows) == 2  # the two reachable points were still written
    assert len(failed) == 1
    assert failed[0]["permanent"] == "True"
    assert "(0.0, -140.0)" == failed[0]["point"]


def test_era5_flow_records_the_failure_in_the_flow_manifest(
    root_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Logged is not enough: a half-successful run has to stay answerable later.
    golden = pd.read_parquet(_ERA5_BRONZE_GOLDEN)
    dead = (0.0, -140.0)
    monkeypatch.setattr(
        era5_bronze, "_fetch_point_table", _failing_era5_fetch(golden, dead, {})
    )

    ingest_era5(request=_era5_request(((37.42, -122.23), dead)), root_uri=root_uri)

    flow = query_flow_manifests(root_uri=root_uri, status=FlowStatus.COMPLETED)
    recorded = flow[0].metadata["failed_points"]
    assert len(recorded) == 1
    assert "not produced any data" in recorded[0]["error"]


def test_era5_flow_does_not_retry_a_permanent_failure(
    root_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Retrying a permanently-failing point spends the provider budget the rest
    # of the run needs, so it must be attempted exactly once.
    golden = pd.read_parquet(_ERA5_BRONZE_GOLDEN)
    dead = (0.0, -140.0)
    attempts: dict = {}
    monkeypatch.setattr(
        era5_bronze, "_fetch_point_table", _failing_era5_fetch(golden, dead, attempts)
    )

    ingest_era5(request=_era5_request(((37.42, -122.23), dead)), root_uri=root_uri)

    assert attempts[dead] == 1


def test_era5_flow_still_raises_when_every_point_fails(
    root_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The guard moved to per-run, it did not disappear: no bronze at all is
    # still a failed run, not a quiet success.
    golden = pd.read_parquet(_ERA5_BRONZE_GOLDEN)
    dead = (0.0, -140.0)
    monkeypatch.setattr(
        era5_bronze, "_fetch_point_table", _failing_era5_fetch(golden, dead, {})
    )

    with pytest.raises(Exception, match="no ERA5 bronze produced"):
        ingest_era5(request=_era5_request((dead,)), root_uri=root_uri)


def test_era5_flow_returns_no_failures_on_a_clean_run(
    root_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    golden = pd.read_parquet(_ERA5_BRONZE_GOLDEN)
    monkeypatch.setattr(era5_bronze, "_fetch_point_table", _fake_era5_fetch(golden))
    assert (
        ingest_era5(request=_era5_request(((37.42, -122.23),)), root_uri=root_uri) == []
    )


def test_pipeline_builds_silver_over_a_point_that_failed_permanently(
    root_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # End to end: a point the provider cannot serve is recorded by bronze, and
    # the pipeline goes on to build silver from the rest rather than aborting.
    # The operator does not have to know in advance which points will die.
    dead = (37.7, -122.4)
    era5_sample = pl.read_parquet(_ERA5_SILVER_SAMPLE)
    nsrdb_sample = pl.read_parquet(_NSRDB_SILVER_SAMPLE)

    def era5_fetch(latitude, longitude, *args, **kwargs):
        if (latitude, longitude) == dead:
            raise failures.PermanentFetchError(
                "CDS cannot serve this request: Request has not produced any data."
            )
        return _rows_for(era5_sample, (latitude, longitude)).to_pandas()

    def nsrdb_fetch(latitude, longitude, year, *args, **kwargs):
        if (latitude, longitude) == dead:
            raise failures.PermanentFetchError(
                "NSRDB rejected the request (400): no data"
            )
        return _fake_nsrdb_fetch(nsrdb_sample)(latitude, longitude, year)

    monkeypatch.setattr(era5_bronze, "_fetch_point_table", era5_fetch)
    monkeypatch.setattr(nsrdb_bronze, "_fetch_point_year_table", nsrdb_fetch)

    run_climate_pipeline(
        request=schema.ClimatePipelineRequestArgs(
            points=_SILVER_POINTS, start_date=_SILVER_START, end_date=_SILVER_END
        ),
        root_uri=root_uri,
    )

    silver_rows = query_manifest(dataset_name=silver.DATASET_NAME, root_uri=root_uri)
    # Silver was built despite the dead point -- one entry per node that survived,
    # which is one fewer than the request asked for.
    assert len(silver_rows) == len(_SILVER_POINTS) - 1
    back = silver.read_points_silver(
        schema.ClimatePipelineRequestArgs(
            points=_SILVER_POINTS, start_date=_SILVER_START, end_date=_SILVER_END
        ),
        root_uri,
        allow_missing=True,
    )
    nodes = set(back.select(["era5_latitude", "era5_longitude"]).unique().rows())
    assert dead not in nodes  # ...covering only the points that resolved
    assert nodes


def test_pipeline_still_fails_when_a_gap_is_not_explained(
    root_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A *transient* failure is not a licence to build without the point: it may
    # well succeed next run, and quietly omitting it would hide that.
    dead = (37.7, -122.4)
    era5_sample = pl.read_parquet(_ERA5_SILVER_SAMPLE)
    nsrdb_sample = pl.read_parquet(_NSRDB_SILVER_SAMPLE)

    def era5_fetch(latitude, longitude, *args, **kwargs):
        if (latitude, longitude) == dead:
            raise RuntimeError("connection reset")
        return _rows_for(era5_sample, (latitude, longitude)).to_pandas()

    monkeypatch.setattr(era5_bronze, "_fetch_point_table", era5_fetch)
    monkeypatch.setattr(
        nsrdb_bronze, "_fetch_point_year_table", _fake_nsrdb_fetch(nsrdb_sample)
    )

    with pytest.raises(Exception, match="bronze covering"):
        run_climate_pipeline(
            request=schema.ClimatePipelineRequestArgs(
                points=_SILVER_POINTS, start_date=_SILVER_START, end_date=_SILVER_END
            ),
            root_uri=root_uri,
        )


def test_silver_records_the_points_it_built_without(
    root_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A table built over a gap has to stay distinguishable from a complete one
    # once the log line has rolled away, or nothing downstream can tell that it
    # covers fewer locations than were requested.
    dead = (37.7, -122.4)
    era5_sample = pl.read_parquet(_ERA5_SILVER_SAMPLE)
    nsrdb_sample = pl.read_parquet(_NSRDB_SILVER_SAMPLE)

    def era5_fetch(latitude, longitude, *args, **kwargs):
        if (latitude, longitude) == dead:
            raise failures.PermanentFetchError("Request has not produced any data.")
        return _rows_for(era5_sample, (latitude, longitude)).to_pandas()

    def nsrdb_fetch(latitude, longitude, year, *args, **kwargs):
        if (latitude, longitude) == dead:
            raise failures.PermanentFetchError("NSRDB rejected the request (400)")
        return _fake_nsrdb_fetch(nsrdb_sample)(latitude, longitude, year)

    monkeypatch.setattr(era5_bronze, "_fetch_point_table", era5_fetch)
    monkeypatch.setattr(nsrdb_bronze, "_fetch_point_year_table", nsrdb_fetch)

    run_climate_pipeline(
        request=schema.ClimatePipelineRequestArgs(
            points=_SILVER_POINTS, start_date=_SILVER_START, end_date=_SILVER_END
        ),
        root_uri=root_uri,
    )

    flows = query_flow_manifests(root_uri=root_uri, status=FlowStatus.COMPLETED)
    silver_flow = [f for f in flows if f.flow_name == "era5_nsrdb_silver"]
    assert len(silver_flow) == 1
    recorded = silver_flow[0].metadata["missing_points"]
    assert any("37.7" in entry for entry in recorded)


def test_silver_records_no_gap_when_everything_resolved(
    root_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The key is absent, not an empty list, so a complete build reads as complete.
    monkeypatch.setattr(
        era5_bronze,
        "_fetch_point_table",
        _fake_era5_fetch(pd.read_parquet(_ERA5_BRONZE_GOLDEN)),
    )
    _seed_silver_bronze(root_uri)
    era5_nsrdb_silver(
        request=schema.ClimatePipelineRequestArgs(
            points=_SILVER_POINTS, start_date=_SILVER_START, end_date=_SILVER_END
        ),
        root_uri=root_uri,
    )
    flows = query_flow_manifests(root_uri=root_uri, status=FlowStatus.COMPLETED)
    silver_flow = [f for f in flows if f.flow_name == "era5_nsrdb_silver"]
    assert "missing_points" not in silver_flow[0].metadata


# --------------------------------------------------------------------------- #
# 5. Overlapping the two bronze sources
# --------------------------------------------------------------------------- #


def test_pipeline_overlaps_the_two_bronze_sources(
    root_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The two providers have independent per-account limits, so the pipeline
    # submits both fan-outs before waiting on either. Proven by making each side
    # wait to see the other in flight: run them one after the other and neither
    # can, because the first has finished before the second starts.
    #
    # One point over one year, so there is exactly one ERA5 request and one NSRDB
    # request -- the only way both waits can succeed is across the two sources.
    era5_in_flight = threading.Event()
    nsrdb_in_flight = threading.Event()
    saw_the_other: list[bool] = []

    def era5_fetch(latitude, longitude, *args, **kwargs) -> pd.DataFrame:
        era5_in_flight.set()
        saw_the_other.append(nsrdb_in_flight.wait(timeout=20))
        return _fake_era5_fetch(pd.read_parquet(_ERA5_BRONZE_GOLDEN))(
            latitude, longitude
        )

    def nsrdb_fetch(latitude, longitude, year, *args, **kwargs) -> pl.DataFrame:
        nsrdb_in_flight.set()
        saw_the_other.append(era5_in_flight.wait(timeout=20))
        return _fake_nsrdb_fetch(pl.read_parquet(_NSRDB_BRONZE_GOLDEN))(
            latitude, longitude, year
        )

    monkeypatch.setattr(era5_bronze, "_fetch_point_table", era5_fetch)
    monkeypatch.setattr(nsrdb_bronze, "_fetch_point_year_table", nsrdb_fetch)

    run_climate_pipeline(
        request=schema.ClimatePipelineRequestArgs(
            points=((37.77, -122.42),),
            start_date=datetime.date(2023, 1, 1),
            end_date=datetime.date(2023, 1, 31),
        ),
        root_uri=root_uri,
    )

    assert saw_the_other == [True, True]  # each source met the other mid-flight


def test_pipeline_records_bronze_writes_and_failures_by_source(
    root_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The bronze fetches are the pipeline's own task runs, so there is no
    # per-source flow-manifest row to carry them. One entry keyed by source has to
    # answer what each provider produced -- and stay answerable about a dead point.
    dead = (37.7, -122.4)
    era5_sample = pl.read_parquet(_ERA5_SILVER_SAMPLE)
    nsrdb_sample = pl.read_parquet(_NSRDB_SILVER_SAMPLE)

    def era5_fetch(latitude, longitude, *args, **kwargs):
        if (latitude, longitude) == dead:
            raise failures.PermanentFetchError("Request has not produced any data.")
        return _rows_for(era5_sample, (latitude, longitude)).to_pandas()

    def nsrdb_fetch(latitude, longitude, year, *args, **kwargs):
        if (latitude, longitude) == dead:
            raise failures.PermanentFetchError("NSRDB rejected the request (400)")
        return _fake_nsrdb_fetch(nsrdb_sample)(latitude, longitude, year)

    monkeypatch.setattr(era5_bronze, "_fetch_point_table", era5_fetch)
    monkeypatch.setattr(nsrdb_bronze, "_fetch_point_year_table", nsrdb_fetch)

    run_climate_pipeline(
        request=schema.ClimatePipelineRequestArgs(
            points=_SILVER_POINTS, start_date=_SILVER_START, end_date=_SILVER_END
        ),
        root_uri=root_uri,
    )

    flows = query_flow_manifests(root_uri=root_uri, status=FlowStatus.COMPLETED)
    pipeline = [f for f in flows if f.flow_name == "run_climate_pipeline"]
    assert len(pipeline) == 1
    written = pipeline[0].metadata["output_write_ids"]
    failed = pipeline[0].metadata["failed_points"]
    # Both sources are named, and the dead point is recorded against each.
    assert set(written) == set(failed) == {"era5", "nsrdb"}
    assert written["era5"] and written["nsrdb"]
    for source in ("era5", "nsrdb"):
        assert [e["point"] for e in failed[source]] == [f"({dead[0]}, {dead[1]})"]
        assert failed[source][0]["permanent"] == "True"

    # No rows are invented for bronze flow runs that never happened: a pipeline run
    # records itself and the silver subflow, nothing more.
    assert {f.flow_name for f in flows} == {
        "run_climate_pipeline",
        "era5_nsrdb_silver",
    }


# --------------------------------------------------------------------------- #
# 6. A point-year the provider cannot serve
# --------------------------------------------------------------------------- #

_MULTI_YEAR_START = datetime.date(2021, 1, 1)
_MULTI_YEAR_END = datetime.date(2023, 12, 31)


def _nsrdb_years_seen(calls: list[int]):
    """A fake NSRDB fetch that records every year it was asked for."""
    golden = pl.read_parquet(_NSRDB_BRONZE_GOLDEN)
    inner = _fake_nsrdb_fetch(golden)

    def _fetch(latitude, longitude, year, *args, **kwargs) -> pl.DataFrame:
        calls.append(year)
        return inner(latitude, longitude, year)

    return _fetch


def test_nsrdb_keeps_the_years_it_got_when_a_later_year_is_permanently_gone(
    root_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Those requests are already spent, and at one request per second discarding
    # two good years to report a third as missing is the expensive way to learn
    # nothing. The write claims only the span it holds.
    calls: list[int] = []
    good = _nsrdb_years_seen(calls)

    def nsrdb_fetch(latitude, longitude, year, *args, **kwargs) -> pl.DataFrame:
        if year == 2023:
            raise failures.PermanentFetchError("NSRDB rejected the request (400)")
        return good(latitude, longitude, year)

    monkeypatch.setattr(nsrdb_bronze, "_fetch_point_year_table", nsrdb_fetch)

    failed = ingest_nsrdb(
        request=schema.ClimatePipelineRequestArgs(
            points=((37.77, -122.42),),
            start_date=_MULTI_YEAR_START,
            end_date=_MULTI_YEAR_END,
        ),
        root_uri=root_uri,
    )

    # 2021 + 2022 were kept; the dead year stopped the loop rather than the point.
    rows = query_manifest(dataset_name=nsrdb_bronze.DATASET_NAME, root_uri=root_uri)
    assert len(rows) == 1
    key = schema.NsrdbPointSliceKey.model_validate_json(rows[0].params_json)
    assert key.end_date == datetime.date(2022, 12, 31)
    assert key.start_date == _MULTI_YEAR_START

    # Nothing is fetched past the permanent failure.
    assert calls == [2021, 2022]

    # ...and the shortfall is recorded, not merely logged.
    assert len(failed) == 1
    assert failed[0]["permanent"] == "True"
    assert "2022-12-31" in failed[0]["error"]


def test_nsrdb_narrowed_write_is_not_reused_for_the_full_range(
    root_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The whole reason for narrowing: a short entry must read as partial coverage,
    # so a later run looks for the rest instead of skipping the point.
    def nsrdb_fetch(latitude, longitude, year, *args, **kwargs) -> pl.DataFrame:
        if year == 2023:
            raise failures.PermanentFetchError("NSRDB rejected the request (400)")
        return _nsrdb_years_seen([])(latitude, longitude, year)

    monkeypatch.setattr(nsrdb_bronze, "_fetch_point_year_table", nsrdb_fetch)
    request = schema.ClimatePipelineRequestArgs(
        points=((37.77, -122.42),),
        start_date=_MULTI_YEAR_START,
        end_date=_MULTI_YEAR_END,
    )
    ingest_nsrdb(request=request, root_uri=root_uri, writer="w")

    # The stored entry does not cover the request, so the point is offered again.
    to_fetch, reusable = point_manifest.split_by_coverage(
        [(37.77, -122.42)],
        nsrdb_bronze.DATASET_NAME,
        root_uri,
        point_manifest.read_point_key,
        request.start_date,
        request.end_date,
    )
    assert to_fetch == [(37.77, -122.42)]
    assert reusable == []


def test_nsrdb_retries_the_point_when_a_year_fails_transiently(
    root_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A transient gap must not be narrowed: that data is expected to arrive, and
    # recording it as a final range would stop any later run looking for it. So the
    # point fails, the task retries, and the retry gets the whole range.
    calls: list[int] = []
    good = _nsrdb_years_seen(calls)
    attempts: list[int] = []

    def nsrdb_fetch(latitude, longitude, year, *args, **kwargs) -> pl.DataFrame:
        if year == 2023:
            attempts.append(year)
            if len(attempts) == 1:  # fail once, transiently, on the last year
                raise RuntimeError("connection reset")
        return good(latitude, longitude, year)

    monkeypatch.setattr(nsrdb_bronze, "_fetch_point_year_table", nsrdb_fetch)

    failed = ingest_nsrdb(
        request=schema.ClimatePipelineRequestArgs(
            points=((37.77, -122.42),),
            start_date=_MULTI_YEAR_START,
            end_date=_MULTI_YEAR_END,
        ),
        root_uri=root_uri,
    )

    assert failed == []
    rows = query_manifest(dataset_name=nsrdb_bronze.DATASET_NAME, root_uri=root_uri)
    key = schema.NsrdbPointSliceKey.model_validate_json(rows[-1].params_json)
    assert key.end_date == _MULTI_YEAR_END  # full range, not narrowed


def test_nsrdb_fails_the_point_when_its_first_year_is_permanently_gone(
    root_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Nothing salvageable: there is no honest range to claim, so the point fails
    # outright and no entry is written for it.
    monkeypatch.setattr(
        nsrdb_bronze,
        "_fetch_point_year_table",
        lambda *a, **k: (_ for _ in ()).throw(
            failures.PermanentFetchError("NSRDB rejected the request (400)")
        ),
    )

    with pytest.raises(RuntimeError, match="no NSRDB bronze produced"):
        ingest_nsrdb(
            request=schema.ClimatePipelineRequestArgs(
                points=((37.77, -122.42),),
                start_date=_MULTI_YEAR_START,
                end_date=_MULTI_YEAR_END,
            ),
            root_uri=root_uri,
        )
    assert not query_manifest(dataset_name=nsrdb_bronze.DATASET_NAME, root_uri=root_uri)


# --------------------------------------------------------------------------- #
# 7. Dead-lettering: what an earlier run recorded, a later run acts on
# --------------------------------------------------------------------------- #


def test_a_point_recorded_dead_is_not_re_attempted(
    root_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Recording a permanent failure and then re-asking for it every run spends the
    # provider's budget forever on an answer already given. The second run must not
    # touch the dead point -- while still fetching a genuinely new one.
    golden = pd.read_parquet(_ERA5_BRONZE_GOLDEN)
    dead = (0.0, -140.0)
    live = (37.42, -122.23)
    added = (40.72, -73.96)

    first: dict = {}
    monkeypatch.setattr(
        era5_bronze, "_fetch_point_table", _failing_era5_fetch(golden, dead, first)
    )
    ingest_era5(request=_era5_request((live, dead)), root_uri=root_uri)
    assert first[dead] == 1  # asked once, refused

    second: dict = {}
    monkeypatch.setattr(
        era5_bronze, "_fetch_point_table", _failing_era5_fetch(golden, dead, second)
    )
    failed = ingest_era5(request=_era5_request((live, dead, added)), root_uri=root_uri)

    assert dead not in second  # never asked again
    assert added in second  # ...but the new point still is
    # Still reported, so the gap stays visible rather than silently vanishing.
    assert [e["point"] for e in failed] == [f"({dead[0]}, {dead[1]})"]
    assert failed[0]["permanent"] == "True"


def test_force_refresh_re_attempts_a_point_recorded_dead(
    root_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The way back for a point judged dead in error, or one the provider has since
    # started serving: suppression must never be a one-way door.
    golden = pd.read_parquet(_ERA5_BRONZE_GOLDEN)
    dead = (0.0, -140.0)
    live = (37.42, -122.23)

    monkeypatch.setattr(
        era5_bronze, "_fetch_point_table", _failing_era5_fetch(golden, dead, {})
    )
    ingest_era5(request=_era5_request((live, dead)), root_uri=root_uri)

    retried: dict = {}
    monkeypatch.setattr(
        era5_bronze, "_fetch_point_table", _failing_era5_fetch(golden, dead, retried)
    )
    ingest_era5(
        request=_era5_request((live, dead)), root_uri=root_uri, force_refresh=True
    )
    assert retried[dead] == 1


def test_silver_alone_builds_over_a_gap_an_earlier_run_explained(
    root_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Run standalone, silver used to have no way to tell an expected gap from
    # bronze never ingested, so it refused until told otherwise by hand -- and that
    # flag waives the check for every gap at once. It now reads the record instead.
    dead = (37.7, -122.4)
    era5_sample = pl.read_parquet(_ERA5_SILVER_SAMPLE)
    nsrdb_sample = pl.read_parquet(_NSRDB_SILVER_SAMPLE)

    def era5_fetch(latitude, longitude, *args, **kwargs):
        if (latitude, longitude) == dead:
            raise failures.PermanentFetchError("Request has not produced any data.")
        return _rows_for(era5_sample, (latitude, longitude)).to_pandas()

    def nsrdb_fetch(latitude, longitude, year, *args, **kwargs):
        if (latitude, longitude) == dead:
            raise failures.PermanentFetchError("NSRDB rejected the request (400)")
        return _fake_nsrdb_fetch(nsrdb_sample)(latitude, longitude, year)

    monkeypatch.setattr(era5_bronze, "_fetch_point_table", era5_fetch)
    monkeypatch.setattr(nsrdb_bronze, "_fetch_point_year_table", nsrdb_fetch)

    request = schema.ClimatePipelineRequestArgs(
        points=_SILVER_POINTS, start_date=_SILVER_START, end_date=_SILVER_END
    )
    # Bronze first, as its own runs -- the shape the deployments actually run.
    ingest_era5(request=request, root_uri=root_uri)
    ingest_nsrdb(request=request, root_uri=root_uri)

    # ...then silver on its own, with no allow_missing_points.
    era5_nsrdb_silver(request=request, root_uri=root_uri)

    back = silver.read_points_silver(request, root_uri, allow_missing=True)
    nodes = set(back.select(["era5_latitude", "era5_longitude"]).unique().rows())
    assert nodes and dead not in nodes


def test_silver_alone_still_refuses_a_gap_nothing_recorded(
    root_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The gap that matters: bronze was simply never ingested for that point. No run
    # ever explained it, so building over it would quietly ship a smaller grid.
    monkeypatch.setattr(
        era5_bronze,
        "_fetch_point_table",
        _fake_era5_fetch(pd.read_parquet(_ERA5_BRONZE_GOLDEN)),
    )
    _seed_silver_bronze(root_uri)

    wider = schema.ClimatePipelineRequestArgs(
        points=(*_SILVER_POINTS, (34.05, -118.24)),  # never ingested, never refused
        start_date=_SILVER_START,
        end_date=_SILVER_END,
    )
    with pytest.raises(Exception, match="bronze covering"):
        era5_nsrdb_silver(request=wider, root_uri=root_uri)


def _nsrdb_hours(points: int, years: int = 27) -> float:
    return points * years / flows.NSRDB_SUSTAINED_REQUESTS_PER_SECOND / 3600


def test_the_sustained_rate_comes_from_the_published_budget() -> None:
    # Not a guess and not the slot-decay figure: the decay rate is how the cap is
    # expressed, this is the rate it sustains. 1,000 requests/hour is what NSRDB
    # publishes, so wall-clock estimates have to be built on that.
    assert nsrdb_bronze.DOCUMENTED_REQUESTS_PER_HOUR == 1_000
    assert flows.NSRDB_SUSTAINED_REQUESTS_PER_SECOND == pytest.approx(1_000 / 3600)


def test_bronze_timeout_covers_a_supervisable_run_not_the_largest_one() -> None:
    # The honest shape of the constraint: at 1,000 requests/hour a few hundred
    # points of full history fits in one run and thousands cannot, whatever the
    # timeout says. Pinning both halves so neither is quietly "fixed" by inflating
    # the number -- the answer for a bigger backfill is more runs, not a longer one.
    assert _nsrdb_hours(500) < flows.BRONZE_TIMEOUT_SECONDS / 3600
    assert _nsrdb_hours(2_000) > flows.PIPELINE_TIMEOUT_SECONDS / 3600
    # The pipeline overlaps the sources, so it needs the larger plus silver.
    assert flows.PIPELINE_TIMEOUT_SECONDS == (
        flows.BRONZE_TIMEOUT_SECONDS + flows.SILVER_TIMEOUT_SECONDS
    )


# --------------------------------------------------------------------------- #
# 8. The provider budgets are actually held around each request
# --------------------------------------------------------------------------- #


def _budget_recorder(events: list[str]):
    """Stand-ins for the Prefect limits that record when they are entered/left."""

    @contextlib.contextmanager
    def fake_concurrency(name, **kwargs):
        events.append(f"enter {name}")
        try:
            yield
        finally:
            events.append(f"leave {name}")

    def fake_rate_limit(name, **kwargs):
        events.append(f"rate {name}")

    return fake_concurrency, fake_rate_limit


def test_era5_fetch_runs_inside_the_cds_limit(
    root_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The limits are server-side, so they are no-ops offline -- which means
    # nothing else in this suite would notice if the wrappers were dropped in a
    # refactor and every fetch started ignoring the provider's budget. Pin that
    # the request happens *inside* the limit, which is the actual requirement.
    events: list[str] = []
    fake_concurrency, fake_rate_limit = _budget_recorder(events)
    monkeypatch.setattr(flows, "concurrency", fake_concurrency)
    monkeypatch.setattr(flows, "rate_limit", fake_rate_limit)

    golden = pd.read_parquet(_ERA5_BRONZE_GOLDEN)
    inner = _fake_era5_fetch(golden)

    def fetch(latitude, longitude, *args, **kwargs):
        events.append("fetch")
        return inner(latitude, longitude)

    monkeypatch.setattr(era5_bronze, "_fetch_point_table", fetch)

    ingest_era5(request=_era5_request(((37.42, -122.23),)), root_uri=root_uri)

    assert events == [f"enter {flows.CDS_LIMIT}", "fetch", f"leave {flows.CDS_LIMIT}"]


def test_nsrdb_fetch_runs_inside_both_the_count_and_rate_limits(
    root_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # NSRDB needs two caps -- how many may be in flight and how fast they start --
    # and the rate limit has to be taken inside the concurrency slot, per year.
    events: list[str] = []
    fake_concurrency, fake_rate_limit = _budget_recorder(events)
    monkeypatch.setattr(flows, "concurrency", fake_concurrency)
    monkeypatch.setattr(flows, "rate_limit", fake_rate_limit)

    inner = _fake_nsrdb_fetch(pl.read_parquet(_NSRDB_BRONZE_GOLDEN))

    def fetch(latitude, longitude, year, *args, **kwargs):
        events.append("fetch")
        return inner(latitude, longitude, year)

    monkeypatch.setattr(nsrdb_bronze, "_fetch_point_year_table", fetch)

    ingest_nsrdb(
        request=schema.ClimatePipelineRequestArgs(
            points=((37.77, -122.42),),
            start_date=datetime.date(2022, 1, 1),
            end_date=datetime.date(2023, 12, 31),
        ),
        root_uri=root_uri,
    )

    # One full cycle per requested year, the rate limit taken within the slot.
    one_year = [
        f"enter {flows.NSRDB_LIMIT}",
        f"rate {flows.NSRDB_RATE_LIMIT}",
        "fetch",
        f"leave {flows.NSRDB_LIMIT}",
    ]
    assert events == one_year * 2


def test_a_point_with_no_values_is_recorded_not_written(
    root_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # End to end for the ocean-point case: ERA5-Land answers with timestamps and
    # no measurements. The point must be recorded as permanently failed and left
    # unwritten -- storing it would mark the range covered forever and hand silver
    # a node with nothing in it.
    golden = pd.read_parquet(_ERA5_BRONZE_GOLDEN)
    at_sea = (34.5, -122.0)
    inner = _fake_era5_fetch(golden)

    def fetch(latitude, longitude, *args, **kwargs) -> pd.DataFrame:
        table = inner(latitude, longitude)
        if (latitude, longitude) == at_sea:
            for column in era5_bronze.VALUE_COLUMNS:
                table[column] = pd.NA
        return table

    monkeypatch.setattr(era5_bronze, "_fetch_point_table", fetch)

    failed = ingest_era5(
        request=_era5_request(((37.42, -122.23), at_sea)), root_uri=root_uri
    )

    # Only the land point was written.
    rows = query_manifest(dataset_name=era5_bronze.DATASET_NAME, root_uri=root_uri)
    assert len(rows) == 1
    written = schema.Era5PointSliceKey.model_validate_json(rows[0].params_json)
    assert written.point == (37.42, -122.23)

    # ...and the sea point is recorded as permanent, so it is not retried now and
    # not re-asked on the next run.
    assert [e["point"] for e in failed] == [f"({at_sea[0]}, {at_sea[1]})"]
    assert failed[0]["permanent"] == "True"
    assert "no values" in failed[0]["error"]


def test_the_plan_log_names_the_points_it_reuses(
    root_uri: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A count alone leaves the question you actually have unanswered: was *my*
    # point reused, or quietly left out? Naming them answers it without a dig
    # through the manifest.
    golden = pd.read_parquet(_ERA5_BRONZE_GOLDEN)
    monkeypatch.setattr(era5_bronze, "_fetch_point_table", _fake_era5_fetch(golden))
    first, second, added = (37.42, -122.23), (40.72, -73.96), (34.03, -118.22)

    ingest_era5(request=_era5_request((first, second)), root_uri=root_uri)

    with caplog.at_level("INFO"):
        ingest_era5(request=_era5_request((first, second, added)), root_uri=root_uri)

    # Both reused points appear by coordinate *and* by the range their stored
    # write covers -- a point reused from a wider earlier write and one covered
    # exactly are otherwise indistinguishable. Named as *requested*, not as the
    # node they matched at: the coordinate the caller asked for is the one they
    # can recognise.
    span = "2025-06-01..2025-06-01"  # the range _era5_request covers
    # The requested range leads the line and applies throughout; each reused
    # point then carries the range its own stored write covers.
    assert f"ERA5 {span}:" in caplog.text
    assert "2 already covered" in caplog.text
    assert f"(37.42, -122.23) {span}" in caplog.text
    assert f"(40.72, -73.96) {span}" in caplog.text
    # The point being fetched is named too, not just counted.
    assert "fetching 1 [(34.03, -118.22)]" in caplog.text


def test_the_plan_log_names_points_held_back_as_dead(
    root_uri: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The count that used to hide this was the confusing one: a suppressed point
    # vanished from the plan entirely, so a two-point request read as one.
    golden = pd.read_parquet(_ERA5_BRONZE_GOLDEN)
    dead = (0.0, -140.0)
    monkeypatch.setattr(
        era5_bronze, "_fetch_point_table", _failing_era5_fetch(golden, dead, {})
    )
    ingest_era5(request=_era5_request(((37.42, -122.23), dead)), root_uri=root_uri)

    with caplog.at_level("INFO"):
        ingest_era5(request=_era5_request(((37.42, -122.23), dead)), root_uri=root_uri)

    assert "1 recorded dead" in caplog.text
    assert f"({dead[0]}, {dead[1]})" in caplog.text


def test_the_plan_log_names_points_on_a_first_run(
    root_uri: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A first run reuses nothing, so the "already covered" clause is absent -- the
    # range and the points still have to be on the line, or the commonest case of
    # all reports only a bare count.
    golden = pd.read_parquet(_ERA5_BRONZE_GOLDEN)
    monkeypatch.setattr(era5_bronze, "_fetch_point_table", _fake_era5_fetch(golden))

    with caplog.at_level("INFO"):
        ingest_era5(
            request=_era5_request(((34.1, -118.2), (45.5, -122.7))), root_uri=root_uri
        )

    assert (
        "ERA5 2025-06-01..2025-06-01: fetching 2 [(34.1, -118.2), (45.5, -122.7)]"
        in (caplog.text)
    )
    assert "already covered" not in caplog.text


def test_nsrdb_bronze_flow_reuses_when_the_attribute_set_narrows(
    root_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    golden = pl.read_parquet(_NSRDB_BRONZE_GOLDEN)
    inner = _fake_nsrdb_fetch(golden)
    calls: list = []

    def _counting(latitude, longitude, year, *args, **kwargs):
        calls.append((latitude, longitude, year))
        return inner(latitude, longitude, year, *args, **kwargs)

    monkeypatch.setattr(nsrdb_bronze, "_fetch_point_year_table", _counting)
    request = schema.ClimatePipelineRequestArgs(
        points=((37.77, -122.42),),
        start_date=datetime.date(2023, 1, 1),
        end_date=datetime.date(2023, 1, 31),
    )
    ingest_nsrdb(request=request, root_uri=root_uri)

    # The stored write holds every curated attribute, so it answers a request for
    # two of them: dropping attributes costs nothing.
    calls.clear()
    ingest_nsrdb(
        request=request,
        nsrdb=schema.NsrdbRequestArgs(attributes=("ghi", "dni")),
        root_uri=root_uri,
    )
    assert calls == []


def test_nsrdb_bronze_flow_refetches_when_the_attribute_set_widens(
    root_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    golden = pl.read_parquet(_NSRDB_BRONZE_GOLDEN)
    inner = _fake_nsrdb_fetch(golden)
    calls: list = []

    def _counting(latitude, longitude, year, nsrdb, *args, **kwargs):
        calls.append((latitude, longitude, year))
        table = inner(latitude, longitude, year)
        return table.select(
            nsrdb_bronze.KEY_COLUMNS + nsrdb_bronze.selected_columns(nsrdb)
        )

    monkeypatch.setattr(nsrdb_bronze, "_fetch_point_year_table", _counting)
    request = schema.ClimatePipelineRequestArgs(
        points=((37.77, -122.42),),
        start_date=datetime.date(2023, 1, 1),
        end_date=datetime.date(2023, 1, 31),
    )
    ingest_nsrdb(
        request=request,
        nsrdb=schema.NsrdbRequestArgs(attributes=("ghi",)),
        root_uri=root_uri,
    )

    # The stored write holds one attribute, so it cannot answer a request for the
    # whole set -- without the recorded selection this point would read as
    # covered and the other thirteen attributes would stay null forever.
    calls.clear()
    ingest_nsrdb(request=request, root_uri=root_uri)
    assert len(calls) == 1


def test_a_30_minute_write_is_not_coverage_for_a_60_minute_request(
    root_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # interval is part of NSRDB's dataset identity, and every reader filters on it.
    # A planner that does not would report the 30-minute write as coverage, never
    # fetch the 60-minute data, and leave silver -- which resolves at 60 -- with
    # nothing. Re-running would not recover, since the plan keeps saying "covered".
    golden = pl.read_parquet(_NSRDB_BRONZE_GOLDEN)
    monkeypatch.setattr(
        nsrdb_bronze, "_fetch_point_year_table", _fake_nsrdb_fetch(golden)
    )
    request = schema.ClimatePipelineRequestArgs(
        points=((37.77, -122.42),),
        start_date=datetime.date(2023, 1, 1),
        end_date=datetime.date(2023, 1, 31),
    )

    ingest_nsrdb(
        request=request,
        nsrdb=schema.NsrdbRequestArgs(interval=30),
        root_uri=root_uri,
    )
    ingest_nsrdb(
        request=request,
        nsrdb=schema.NsrdbRequestArgs(interval=60),
        root_uri=root_uri,
    )

    # Two entries, one per interval -- not one 30-minute entry reused for both.
    intervals = sorted(
        schema.NsrdbPointSliceKey.model_validate_json(row.params_json).interval
        for row in query_manifest(
            dataset_name=nsrdb_bronze.DATASET_NAME, root_uri=root_uri
        )
    )
    assert intervals == [30, 60]

    # ...and the 60-minute slice silver reads actually resolves.
    back = nsrdb_bronze.read_points_bronze(
        request, schema.NsrdbRequestArgs(interval=60), root_uri
    )
    assert back.height > 0
