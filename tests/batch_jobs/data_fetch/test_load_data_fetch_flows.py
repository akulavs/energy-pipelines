"""
Tests for batch_jobs.data_fetch.load -- the orchestration half of the load helper.

The question here is whether the pipeline runs, what it is handed, and that the
frames come back. The pipeline itself is covered by tests/batch_jobs/load_pipeline,
so ``run_load_pipeline`` is the seam these mock: this module's job is deciding
*whether* to call it, not what it does.

The store is seeded from the committed load fixtures through the normal write
path, so the coverage the helper reads is coverage the pipeline would have
produced.
"""

from __future__ import annotations

import pathlib

import polars as pl
import pytest

from batch_jobs.data_fetch import load as data_fetch_flow
from common.exceptions import PipelineValueError
from external_data.load_pipeline import schema, silver
from external_data.load_pipeline.comstock import bronze as comstock_bronze
from external_data.load_pipeline.dsgrid import bronze as dsgrid_bronze
from external_data.load_pipeline.resstock import bronze as resstock_bronze

_FIXTURES = (
    pathlib.Path(__file__).parents[3]
    / "packages/external-data/tests/load_pipeline/silver/fixtures"
)

# The geography the committed fixtures cover.
_PUMA = "G11000101"
_STATE = "DC"
# A second DC PUMA, used only where a slice must be missing.
_OTHER_PUMA = "G11000105"


def _fixture(name: str) -> pl.DataFrame:
    return pl.read_parquet(_FIXTURES / f"{name}.parquet")


@pytest.fixture
def root_uri(tmp_path: pathlib.Path) -> str:
    return str(tmp_path / "load_pipeline")


def _collapsed_puma_metadata() -> pl.DataFrame:
    """The county metadata collapsed the way the per-PUMA file is published."""
    county = _fixture("comstock_metadata_bronze")
    carried = [
        c
        for c in comstock_bronze.PUMA_METADATA_COLUMNS
        if c not in ("bldg_id", "weight")
    ]
    return (
        county.group_by("bldg_id")
        .agg(pl.col("weight").sum(), *[pl.col(c).first() for c in carried])
        .select(comstock_bronze.PUMA_METADATA_COLUMNS)
        .sort("bldg_id")
    )


def _seed(root: str, puma: str = _PUMA) -> None:
    """
    Write every dataset a run would produce for *puma*, through the normal path.

    Bronze from the committed fixtures; silver built from that bronze, which is
    local work and needs no network.
    """
    geography = schema.LoadGeography(puma_gisjoin=puma)
    resstock_bronze.write_metadata_bronze(
        _fixture("resstock_metadata_bronze"),
        silver.resstock_metadata_request(geography),
        root,
        writer="seed",
    )
    resstock_bronze.write_puma_timeseries_bronze(
        _fixture("resstock_timeseries_bronze"),
        silver.resstock_timeseries_request(geography),
        root,
        writer="seed",
    )
    comstock_bronze.write_puma_metadata_bronze(
        _collapsed_puma_metadata(),
        silver.comstock_puma_metadata_request(geography),
        root,
        writer="seed",
    )
    comstock_bronze.write_puma_timeseries_bronze(
        _fixture("comstock_timeseries_bronze"),
        silver.comstock_timeseries_request(geography),
        root,
        writer="seed",
    )
    for request in silver.dsgrid_requests(geography.state):
        profile = dsgrid_bronze.PROFILES[request.source]
        dsgrid_bronze.write_metadata_bronze(
            _fixture(profile.metadata_dataset_name), request, root, writer="seed"
        )
        dsgrid_bronze.write_timeseries_bronze(
            _fixture(profile.timeseries_dataset_name), request, root, writer="seed"
        )
    silver.ingest_industrial_silver(
        schema.IndustrialLoadRequestArgs(state=geography.state),
        bronze_root=root,
        silver_root=root,
        writer="seed",
    )


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Replace every OEDI network seam with the committed fixtures, so a test can
    drive a real ``run_load_pipeline`` rather than mock it.

    The building-stock fakes ignore which PUMA is asked for and return the
    fixture's buildings: what these tests count is *how many silver writes a run
    makes*, not which buildings landed where.
    """
    resstock_ts = _fixture("resstock_timeseries_bronze")
    resstock_meta = _fixture("resstock_metadata_bronze")
    comstock_ts = _fixture("comstock_timeseries_bronze")
    puma_metadata = _collapsed_puma_metadata()
    dsgrid_tables = {
        source: (
            _fixture(dsgrid_bronze.PROFILES[source].metadata_dataset_name),
            _fixture(dsgrid_bronze.PROFILES[source].timeseries_dataset_name),
        )
        for source in dsgrid_bronze.PROFILES
    }

    monkeypatch.setattr(resstock_bronze, "download_object", lambda *a, **k: b"")
    monkeypatch.setattr(
        resstock_bronze, "parquet_to_metadata_table", lambda *a, **k: resstock_meta
    )
    monkeypatch.setattr(
        resstock_bronze,
        "resolve_puma_building_ids",
        lambda *a, **k: sorted(resstock_ts["bldg_id"].unique().to_list()),
    )
    monkeypatch.setattr(
        resstock_bronze,
        "fetch_building_frame",
        lambda args, bldg_id, client: resstock_ts.filter(pl.col("bldg_id") == bldg_id),
    )
    monkeypatch.setattr(comstock_bronze, "download_object", lambda *a, **k: b"")
    monkeypatch.setattr(
        comstock_bronze,
        "parquet_to_puma_metadata_table",
        lambda *a, **k: puma_metadata,
    )
    monkeypatch.setattr(
        comstock_bronze,
        "fetch_building_frame",
        lambda args, bldg_id, client: comstock_ts.filter(pl.col("bldg_id") == bldg_id),
    )
    monkeypatch.setattr(dsgrid_bronze.dsg_common, "load_dsg_bytes", lambda *a, **k: b"")
    monkeypatch.setattr(
        dsgrid_bronze,
        "reconstruct_tables",
        lambda data, state, source=dsgrid_bronze.DsgridSource.INDUSTRIAL: dsgrid_tables[
            source
        ],
    )


def _ingest(root_uri: str, pumas=(_PUMA,), **kwargs):
    return data_fetch_flow.fetch_and_ingest(pumas, root_uri=root_uri, **kwargs)


def _fetch(root_uri: str, pumas=(_PUMA,), **kwargs):
    return data_fetch_flow.fetch(pumas, root_uri=root_uri, **kwargs)


# --------------------------------------------------------------------------- #
# Whether the pipeline runs
# --------------------------------------------------------------------------- #


def test_a_covered_request_skips_the_pipeline_and_returns_every_dataset(
    root_uri: str, mocker
) -> None:
    _seed(root_uri)
    flow = mocker.patch.object(
        data_fetch_flow, "run_load_pipeline", side_effect=AssertionError
    )

    result = _fetch(root_uri)

    flow.assert_not_called()
    assert not result.ran_pipeline
    assert result.coverage.complete
    # The six a caller asked for, as named frames. The dsgrid bronze the silver
    # is built from is not among them -- it is an input to the join.
    assert result.resstock_metadata.height > 0
    assert result.resstock_timeseries.height > 0
    assert result.comstock_metadata.height > 0
    assert result.comstock_timeseries.height > 0
    assert result.industrial_metadata.height > 0
    assert result.industrial_timeseries.height > 0


def test_a_missing_puma_runs_the_pipeline(root_uri: str, mocker) -> None:
    _seed(root_uri)
    flow = mocker.patch.object(data_fetch_flow, "run_load_pipeline")

    result = _ingest(root_uri, pumas=(_PUMA, _OTHER_PUMA), allow_missing=True)

    assert flow.call_count == 1
    assert result.ran_pipeline
    # The seeded PUMA's slices are covered; the new PUMA's are not. Its state is
    # already covered, so only the building-stock slices are missing.
    missing = {entry.dataset_name for entry in result.coverage.missing}
    assert missing == {
        resstock_bronze.METADATA_DATASET_NAME,
        resstock_bronze.TIMESERIES_DATASET_NAME,
        comstock_bronze.PUMA_METADATA_DATASET_NAME,
        comstock_bronze.TIMESERIES_DATASET_NAME,
    }


def test_force_refresh_runs_even_when_everything_is_covered(
    root_uri: str, mocker
) -> None:
    _seed(root_uri)
    flow = mocker.patch.object(data_fetch_flow, "run_load_pipeline")

    result = _ingest(root_uri, force_refresh=True)

    assert result.coverage.complete
    assert result.ran_pipeline
    assert flow.call_count == 1


# --------------------------------------------------------------------------- #
# fetch: the read-only door
# --------------------------------------------------------------------------- #


def test_fetch_refuses_when_the_store_has_nothing(root_uri: str, mocker) -> None:
    """
    An empty store has no partial answer to give -- and fetch must not quietly
    start a per-building fan-out over OEDI, which is the whole reason it is a
    separate call.
    """
    flow = mocker.patch.object(
        data_fetch_flow, "run_load_pipeline", side_effect=AssertionError
    )

    with pytest.raises(PipelineValueError, match="has nothing for this request"):
        _fetch(root_uri)

    flow.assert_not_called()


def test_fetch_returns_what_is_there_and_reports_what_is_not(
    root_uri: str, mocker
) -> None:
    _seed(root_uri)
    flow = mocker.patch.object(
        data_fetch_flow, "run_load_pipeline", side_effect=AssertionError
    )

    # One PUMA seeded, one never ingested.
    result = _fetch(root_uri, pumas=(_PUMA, _OTHER_PUMA))

    flow.assert_not_called()
    assert not result.ran_pipeline
    # The seeded PUMA's data still comes back...
    assert result.resstock_timeseries.height > 0
    assert result.industrial_timeseries.height > 0
    # ...and the other is named, per dataset, on the coverage it returns.
    missing = {entry.dataset_name for entry in result.coverage.missing}
    assert resstock_bronze.TIMESERIES_DATASET_NAME in missing


def test_a_fully_covered_state_is_never_handed_to_the_flow(
    root_uri: str, mocker
) -> None:
    """
    The narrowing that does hold: a PUMA with every slice present is left out of
    the request entirely, so the run cannot disturb it.

    ``industrial_load_silver`` rebuilds both tables for every state it is *given*
    -- see the test below for the case that narrowing cannot reach.
    """
    _seed(root_uri)

    captured: list[schema.LoadGeographies] = []
    mocker.patch.object(
        data_fetch_flow,
        "run_load_pipeline",
        side_effect=lambda **kwargs: captured.append(kwargs["geographies"]),
    )

    result = _ingest(root_uri, pumas=(_PUMA, _OTHER_PUMA), allow_missing=True)

    assert result.ran_pipeline
    assert [g.puma_gisjoin for g in captured[0].pumas] == [_OTHER_PUMA]


def test_a_new_puma_still_rebuilds_its_state_silver(
    root_uri: str, offline: None
) -> None:
    """
    The limitation the narrowing cannot reach, recorded rather than asserted away.

    ``industrial_load_silver`` builds both tables for every state in the request
    it is handed. A new PUMA drags its state in, so that state's silver is
    rebuilt even though the industrial tables are dsgrid-derived and owe nothing
    to the building stock -- superseding a good table and orphaning its parquet.

    Fixing it means a per-state coverage check inside the flow, which would also
    help anyone driving it from the Prefect UI. Out of scope here; this test
    fails the day that lands, which is the point.

    Driven through a real run: mocking ``run_load_pipeline`` makes a write count
    trivially equal and proves nothing.
    """
    manifests = pathlib.Path(root_uri) / "_manifests"

    def written(dataset_name: str) -> int:
        return len(list((manifests / dataset_name).glob("*.json")))

    _ingest(root_uri, pumas=(_PUMA,))
    before = written(silver.METADATA_DATASET_NAME)
    assert before == 1

    # A second PUMA of the same state: only its building stock is missing.
    _ingest(root_uri, pumas=(_PUMA, _OTHER_PUMA), allow_missing=True)

    # The state's silver is rebuilt regardless, because the flow was handed a
    # PUMA of that state.
    assert written(silver.METADATA_DATASET_NAME) == before + 1


def test_rows_for_slices_each_frame_by_the_column_it_carries(root_uri: str) -> None:
    """
    The six frames are keyed at two grains -- building stock per PUMA, the
    industrial silver per state -- so slicing by hand means knowing which is
    which. rows_for does it by inspecting the frame.

    Asserted on which PUMA the rows carry rather than on a row count: the
    committed fixtures label every row with the fixture PUMA, so seeding a second
    one produces rows indistinguishable from the first.
    """
    _seed(root_uri)
    result = _fetch(root_uri)

    one = result.rows_for(_PUMA)

    # Building stock filtered by puma_gisjoin, and only that PUMA's rows.
    assert one.resstock_metadata.height > 0
    assert set(one.resstock_metadata["puma_gisjoin"].unique().to_list()) == {_PUMA}
    assert set(one.comstock_metadata["puma_gisjoin"].unique().to_list()) == {_PUMA}
    # The industrial silver is keyed by state, so it is not filtered away.
    assert one.industrial_metadata.height == result.industrial_metadata.height
    assert one.industrial_timeseries.height == result.industrial_timeseries.height


def test_rows_for_rejects_a_puma_the_request_did_not_cover(root_uri: str) -> None:
    _seed(root_uri)
    result = _fetch(root_uri)

    # Names what the request did cover, rather than raising a bare KeyError.
    with pytest.raises(PipelineValueError, match="was not part of this request"):
        result.rows_for("G99999999")
