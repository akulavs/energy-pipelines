"""
Integration tests for batch_jobs.load_pipeline.flows.

The OEDI fetch seams are monkeypatched so the flows run offline, and every frame
they return is a **real** committed bronze slice for Washington, DC (see
``packages/external-data/tests/load_pipeline/silver/fixtures``). Exercises the write +
flow-manifest lifecycle for all five flows; the harmonization correctness is
covered by packages/external-data/tests/load_pipeline.
"""

from __future__ import annotations

import datetime
import importlib
import inspect
import json
import logging
import pathlib
import threading
import time

import polars as pl
import pytest

from batch_jobs.load_pipeline import flows
from batch_jobs.load_pipeline.flows import (
    ingest_comstock,
    ingest_dsgrid,
    ingest_resstock,
    run_load_pipeline,
    industrial_load_silver,
)
from common.exceptions import PipelineError, PipelineValueError
from common.storage import columnar
from common.storage.flow_manifest import (
    FlowStatus,
    query_flow_manifests,
    read_flow_manifest,
)
from common.storage.manifest import query_manifest, resolve_manifest
from external_data.load_pipeline.comstock import bronze as comstock_bronze
from external_data.load_pipeline.dsgrid import bronze as dsgrid_bronze
from external_data.load_pipeline import failures, schema, silver
from external_data.load_pipeline.resstock import bronze as resstock_bronze

# Committed real fixtures from the package tests.
_FIXTURES = (
    pathlib.Path(__file__).parents[3]
    / "packages/external-data/tests/load_pipeline/silver/fixtures"
)

# The geography the fixtures cover.
_STATE = "DC"
_PUMA = "G11000101"
_COUNTY = "G1100010"
_N_INTERVALS = 672
_N_HOURS = 168

INDUSTRIAL = dsgrid_bronze.DsgridSource.INDUSTRIAL
GAPS = dsgrid_bronze.DsgridSource.GAPS


def _fixture(name: str) -> pl.DataFrame:
    return pl.read_parquet(_FIXTURES / f"{name}.parquet")


@pytest.fixture
def root_uri(tmp_path: pathlib.Path) -> str:
    return str(tmp_path / "load_pipeline")


def _geography() -> schema.LoadGeography:
    return schema.LoadGeography(puma_gisjoin=_PUMA)


def _geographies(*puma_gisjoins: str) -> schema.LoadGeographies:
    """What a run is asked for: the fixture PUMA unless others are named."""
    return schema.LoadGeographies.of(*(puma_gisjoins or (_PUMA,)))


def _industrial(state: str = _STATE) -> schema.IndustrialLoadRequestArgs:
    return schema.IndustrialLoadRequestArgs(state=state)


# --------------------------------------------------------------------------- #
# Offline seams
# --------------------------------------------------------------------------- #


@pytest.fixture
def offline_resstock(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Replace ResStock's two network steps with the real committed slices: the
    metadata parse, and the PUMA building-id resolve + per-building assembly.

    The fake assembler filters the fixture to the ``bldg_ids`` it is handed, so the
    ids the flow resolves genuinely reach the assembly step rather than being
    ignored.
    """
    timeseries = _fixture("resstock_timeseries_bronze")
    metadata = _fixture("resstock_metadata_bronze")

    monkeypatch.setattr(resstock_bronze, "download_object", lambda *a, **k: b"")
    monkeypatch.setattr(
        resstock_bronze, "parquet_to_metadata_table", lambda *a, **k: metadata
    )
    monkeypatch.setattr(
        resstock_bronze,
        "resolve_puma_building_ids",
        lambda *a, **k: sorted(timeseries["bldg_id"].unique().to_list()),
    )
    # The fan-out unit is one building now -- the flow maps it, so this is the seam
    # that stands in for the network.
    monkeypatch.setattr(
        resstock_bronze,
        "fetch_building_frame",
        lambda args, bldg_id, client: timeseries.filter(pl.col("bldg_id") == bldg_id),
    )
    monkeypatch.setattr(
        resstock_bronze,
        "fetch_puma_timeseries_table",
        lambda args, client, bldg_ids, **k: timeseries.filter(
            pl.col("bldg_id").is_in(bldg_ids)
        ),
    )


def _collapsed_puma_metadata() -> pl.DataFrame:
    """
    The county bronze slice collapsed the way the per-PUMA file is published.

    One row per building with ``weight`` summed over the census tracts the model
    represents -- 452 fixture rows become 3. Deriving it here rather than
    committing a second parquet keeps the two from drifting apart, and shows the
    collapse rule the ingest depends on.
    """
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


@pytest.fixture
def offline_comstock(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The ComStock network seams: the two metadata parses and the PUMA assembly.

    ``resolve_puma_building_ids`` is deliberately *not* stubbed. The flow reads its
    ids out of the PUMA metadata bronze it just wrote, so if it ever falls back to
    resolving them over the network the unstubbed download surfaces as a failure
    rather than passing quietly.
    """
    timeseries = _fixture("comstock_timeseries_bronze")
    puma_metadata = _collapsed_puma_metadata()

    monkeypatch.setattr(comstock_bronze, "download_object", lambda *a, **k: b"")
    monkeypatch.setattr(
        comstock_bronze,
        "parquet_to_puma_metadata_table",
        lambda *a, **k: puma_metadata,
    )
    monkeypatch.setattr(
        comstock_bronze,
        "fetch_building_frame",
        lambda args, bldg_id, client: timeseries.filter(pl.col("bldg_id") == bldg_id),
    )
    monkeypatch.setattr(
        comstock_bronze,
        "fetch_puma_timeseries_table",
        lambda args, client, bldg_ids, **k: timeseries.filter(
            pl.col("bldg_id").is_in(bldg_ids)
        ),
    )


@pytest.fixture
def offline_dsgrid(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Replace the ``.dsg`` download + HDF5 reconstruction with the real committed
    metadata / timeseries pair for each source file.
    """
    tables = {
        source: (
            _fixture(dsgrid_bronze.PROFILES[source].metadata_dataset_name),
            _fixture(dsgrid_bronze.PROFILES[source].timeseries_dataset_name),
        )
        for source in dsgrid_bronze.PROFILES
    }
    monkeypatch.setattr(dsgrid_bronze.dsg_common, "load_dsg_bytes", lambda *a, **k: b"")
    monkeypatch.setattr(
        dsgrid_bronze,
        "reconstruct_tables",
        lambda data, state, source=INDUSTRIAL: tables[source],
    )


@pytest.fixture
def offline_all(offline_resstock, offline_comstock, offline_dsgrid) -> None:
    return None


# A second PUMA in the same state, served by re-labelling the fixture PUMA's three
# buildings into an id range of their own. The profiles stay the real committed DC
# slices; only the labels differ, which is enough to make a two-PUMA run write two
# genuinely separate sets of keys offline.
_SECOND_PUMA = "G11000105"
_SECOND_PUMA_ID_OFFSET = 100_000


def _relabelled(frame: pl.DataFrame, puma_gisjoin: str) -> pl.DataFrame:
    """The fixture rows as they would read for another PUMA."""
    return frame.with_columns(
        (pl.col("bldg_id") + _SECOND_PUMA_ID_OFFSET).cast(frame.schema["bldg_id"]),
        pl.lit(puma_gisjoin).alias("puma_gisjoin"),
    )


def _building_frame(timeseries: pl.DataFrame, args, bldg_id: int) -> pl.DataFrame:
    """One building's profile, for whichever PUMA asked for it."""
    source_id = bldg_id % _SECOND_PUMA_ID_OFFSET
    return timeseries.filter(pl.col("bldg_id") == source_id).with_columns(
        pl.lit(bldg_id).cast(timeseries.schema["bldg_id"]).alias("bldg_id"),
        pl.lit(args.puma_gisjoin).alias("puma_gisjoin"),
    )


@pytest.fixture
def offline_two_pumas(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """
    The building-stock seams widened to serve ``_SECOND_PUMA`` as well.

    ResStock's metadata is per state, so one frame carries both PUMAs -- which is
    also what lets a two-PUMA run reuse the single metadata write. ComStock's is per
    PUMA, so the seam has to answer per PUMA: the stubbed download hands back the
    URL it was given, and the URL names the PUMA being asked for.
    """
    resstock_timeseries = _fixture("resstock_timeseries_bronze")
    resstock_metadata = _fixture("resstock_metadata_bronze")
    comstock_timeseries = _fixture("comstock_timeseries_bronze")
    comstock_metadata = _collapsed_puma_metadata()
    downloads: list[str] = []

    def _resstock_download(url, **kwargs) -> bytes:
        downloads.append(str(url))
        return b""

    monkeypatch.setattr(resstock_bronze, "download_object", _resstock_download)
    monkeypatch.setattr(
        resstock_bronze,
        "parquet_to_metadata_table",
        lambda *a, **k: pl.concat(
            [resstock_metadata, _relabelled(resstock_metadata, _SECOND_PUMA)]
        ),
    )
    monkeypatch.setattr(
        resstock_bronze,
        "fetch_building_frame",
        lambda args, bldg_id, client: _building_frame(
            resstock_timeseries, args, bldg_id
        ),
    )

    monkeypatch.setattr(
        comstock_bronze, "download_object", lambda url, **k: str(url).encode()
    )
    monkeypatch.setattr(
        comstock_bronze,
        "parquet_to_puma_metadata_table",
        lambda data, *a, **k: (
            comstock_metadata
            if _PUMA in data.decode()
            else _relabelled(comstock_metadata, _SECOND_PUMA)
        ),
    )
    monkeypatch.setattr(
        comstock_bronze,
        "fetch_building_frame",
        lambda args, bldg_id, client: _building_frame(
            comstock_timeseries, args, bldg_id
        ),
    )
    return downloads


# A well-formed GISJOIN in the fixture's state that no release has a file for --
# what an engineer's typo, or a 2020-vintage code against a 2010-vintage release,
# actually looks like. It passes LoadGeography's pattern and its state reads back as
# DC, so nothing catches it before the metadata phase goes looking.
_ABSENT_PUMA = "G11009999"


@pytest.fixture
def offline_one_absent_puma(monkeypatch: pytest.MonkeyPatch, offline_dsgrid) -> None:
    """
    Both building-stock sources serving ``_PUMA`` and refusing ``_ABSENT_PUMA``, each
    the way it really refuses.

    ResStock refuses by omission: its metadata is one file per state, so an absent
    code is simply not among the rows -- the seam returns the DC fixture unchanged
    and ``metadata_for_puma`` finds nothing. ComStock refuses with a 404: its
    metadata is one file *per PUMA*, so the URL for a code it has never published
    does not exist, which ``download_object`` classifies permanent at the raise site.
    """
    resstock_timeseries = _fixture("resstock_timeseries_bronze")
    resstock_metadata = _fixture("resstock_metadata_bronze")
    comstock_timeseries = _fixture("comstock_timeseries_bronze")
    comstock_metadata = _collapsed_puma_metadata()

    # -- ResStock: the state file, which has no such PUMA in it --
    monkeypatch.setattr(resstock_bronze, "download_object", lambda *a, **k: b"")
    monkeypatch.setattr(
        resstock_bronze, "parquet_to_metadata_table", lambda *a, **k: resstock_metadata
    )
    monkeypatch.setattr(
        resstock_bronze,
        "fetch_building_frame",
        lambda args, bldg_id, client: resstock_timeseries.filter(
            pl.col("bldg_id") == bldg_id
        ),
    )

    # -- ComStock: a per-PUMA object that 404s for the absent code --
    def _comstock_download(url, **kwargs) -> bytes:
        if _ABSENT_PUMA in str(url):
            raise failures.PermanentFetchError(
                f"OEDI cannot serve this request (404) for {url}: NoSuchKey"
            )
        return str(url).encode()

    monkeypatch.setattr(comstock_bronze, "download_object", _comstock_download)
    monkeypatch.setattr(
        comstock_bronze,
        "parquet_to_puma_metadata_table",
        lambda *a, **k: comstock_metadata,
    )
    monkeypatch.setattr(
        comstock_bronze,
        "fetch_building_frame",
        lambda args, bldg_id, client: comstock_timeseries.filter(
            pl.col("bldg_id") == bldg_id
        ),
    )


def _rejected(flow) -> list[dict[str, str]]:
    """The rejected-PUMA records on a flow manifest."""
    return (flow.metadata or {}).get(failures.REJECTED_PUMAS_KEY, [])


def _pumas_written(dataset_name: str, root_uri: str) -> set[str]:
    """Which PUMAs a dataset's keys name."""
    return {
        json.loads(row.params_json)["puma_gisjoin"]
        for row in query_manifest(dataset_name=dataset_name, root_uri=root_uri)
    }


def _seed_bronze(root_uri: str) -> schema.LoadGeography:
    """
    Write every bronze slice the silver build reads, directly through the storage
    layer (no flows), for the silver-only tests.
    """
    request = _geography()
    write_time = datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC)
    columnar.write_dataset(
        _fixture("resstock_metadata_bronze"),
        resstock_bronze.ResstockMetadataBronzeSchema,
        resstock_bronze.METADATA_DATASET_NAME,
        silver.resstock_metadata_request(request),
        root_uri,
        writer="seed",
        write_time=write_time,
    )
    columnar.write_dataset(
        _fixture("resstock_timeseries_bronze"),
        resstock_bronze.ResstockTimeseriesBronzeSchema,
        resstock_bronze.TIMESERIES_DATASET_NAME,
        silver.resstock_timeseries_request(request),
        root_uri,
        writer="seed",
        write_time=write_time,
    )
    columnar.write_dataset(
        _fixture("comstock_timeseries_bronze"),
        comstock_bronze.ComstockTimeseriesBronzeSchema,
        comstock_bronze.TIMESERIES_DATASET_NAME,
        silver.comstock_timeseries_request(request),
        root_uri,
        writer="seed",
        write_time=write_time,
    )
    columnar.write_dataset(
        _collapsed_puma_metadata(),
        comstock_bronze.ComstockPumaMetadataBronzeSchema,
        comstock_bronze.PUMA_METADATA_DATASET_NAME,
        silver.comstock_puma_metadata_request(request),
        root_uri,
        writer="seed",
        write_time=write_time,
    )
    for params in silver.dsgrid_requests(_STATE):
        profile = dsgrid_bronze.PROFILES[params.source]
        for name, sch in (
            (profile.timeseries_dataset_name, profile.timeseries_schema),
            (profile.metadata_dataset_name, profile.metadata_schema),
        ):
            columnar.write_dataset(
                _fixture(name),
                sch,
                name,
                params,
                root_uri,
                writer="seed",
                write_time=write_time,
            )
    return request


# --------------------------------------------------------------------------- #
# 1. Residential bronze flow
# --------------------------------------------------------------------------- #


def test_resstock_bronze_flow_writes(root_uri: str, offline_resstock: None) -> None:
    ingest_resstock(geographies=_geographies(), root_uri=root_uri)

    metadata = query_manifest(
        dataset_name=resstock_bronze.METADATA_DATASET_NAME, root_uri=root_uri
    )
    timeseries = query_manifest(
        dataset_name=resstock_bronze.TIMESERIES_DATASET_NAME, root_uri=root_uri
    )
    assert len(metadata) == 1
    assert len(timeseries) == 1
    # The metadata is written first, so a point-in-time read that finds the
    # profiles always finds the weights. It used to be a shared write_time; the
    # ordering took over when the metadata moved above the PUMA loop, and it is
    # the same guarantee -- weights are never the newer of the two.
    assert metadata[0].write_time <= timeseries[0].write_time

    flow = query_flow_manifests(root_uri=root_uri, status=FlowStatus.COMPLETED)
    assert len(flow) == 1
    assert set(flow[0].metadata["output_write_ids"]) == {
        metadata[0].write_id,
        timeseries[0].write_id,
    }


def test_resstock_bronze_flow_writes_every_building_in_the_puma(
    root_uri: str, offline_resstock: None
) -> None:
    """
    A location means all of its buildings.

    There is no cap to honour: ``max_buildings`` was deleted rather than added to the
    key, so no flow argument can narrow what a PUMA write contains. This asserts the
    positive contract -- everything the resolve returned is in the write -- which is
    what makes a short write a bug rather than a supported mode.
    """
    ingest_resstock(geographies=_geographies(), root_uri=root_uri)

    # Read through the helper: one manifest scan matched on the base fields,
    # because the write's key carries a count the caller does not know.
    back = resstock_bronze.read_puma_timeseries_bronze(
        silver.resstock_timeseries_request(_geography()), root_uri
    )
    expected = _fixture("resstock_timeseries_bronze")["bldg_id"].unique().to_list()
    assert sorted(back["bldg_id"].unique().to_list()) == sorted(expected)


# --------------------------------------------------------------------------- #
# 2. Commercial bronze flow
# --------------------------------------------------------------------------- #


def test_comstock_bronze_flow_writes_the_two_dataset_pair(
    root_uri: str, offline_comstock: None
) -> None:
    """
    ComStock mirrors ResStock: one metadata dataset, one timeseries dataset.

    Nothing else -- the release also publishes per-county metadata and pre-weighted
    aggregates, and neither is ingested, so a reader has one table of buildings and
    one of their profiles, joined on bldg_id.
    """
    ingest_comstock(geographies=_geographies(), root_uri=root_uri)

    assert (
        len(
            query_manifest(
                dataset_name=comstock_bronze.PUMA_METADATA_DATASET_NAME,
                root_uri=root_uri,
            )
        )
        == 1
    )
    assert query_manifest(
        dataset_name=comstock_bronze.TIMESERIES_DATASET_NAME, root_uri=root_uri
    )
    # the same two dataset names the catalog registers for ComStock, and no others
    written = {
        row.dataset_name
        for name in (
            comstock_bronze.PUMA_METADATA_DATASET_NAME,
            comstock_bronze.TIMESERIES_DATASET_NAME,
        )
        for row in query_manifest(dataset_name=name, root_uri=root_uri)
    }
    assert written == {
        comstock_bronze.PUMA_METADATA_DATASET_NAME,
        comstock_bronze.TIMESERIES_DATASET_NAME,
    }


def test_comstock_timeseries_uses_the_ids_from_the_puma_metadata_bronze(
    root_uri: str, offline_comstock: None
) -> None:
    """
    The ids reaching the fan-out are the PUMA metadata's, read back from bronze.

    Before this, resolving them downloaded every county metadata file in the state
    -- a second time, since the flow had just ingested them all -- so the cost
    scaled with counties-per-state rather than with the PUMA. ``offline_comstock``
    leaves ``resolve_puma_building_ids`` unstubbed, so a fallback to the network
    would fail here rather than pass quietly.
    """
    ingest_comstock(geographies=_geographies(), root_uri=root_uri)

    written = comstock_bronze.read_puma_timeseries_bronze(
        silver.comstock_timeseries_request(_geography()), root_uri
    )
    puma_metadata = columnar.read_dataset(
        comstock_bronze.ComstockPumaMetadataBronzeSchema,
        comstock_bronze.PUMA_METADATA_DATASET_NAME,
        silver.comstock_puma_metadata_request(_geography()),
        root_uri,
    )
    assert sorted(written["bldg_id"].unique().to_list()) == sorted(
        puma_metadata["bldg_id"].to_list()
    )
    # one row per building: the tract duplication is collapsed at the source
    assert puma_metadata.height == puma_metadata["bldg_id"].n_unique()


# --------------------------------------------------------------------------- #
# 3. Industrial bronze flow
# --------------------------------------------------------------------------- #


def test_dsgrid_bronze_flow_writes_both_sources(
    root_uri: str, offline_dsgrid: None
) -> None:
    ingest_dsgrid(geographies=_geographies(), root_uri=root_uri)

    # Both source files, each with a metadata + timeseries dataset.
    for source in dsgrid_bronze.PROFILES:
        profile = dsgrid_bronze.PROFILES[source]
        assert query_manifest(
            dataset_name=profile.metadata_dataset_name, root_uri=root_uri
        )
        assert query_manifest(
            dataset_name=profile.timeseries_dataset_name, root_uri=root_uri
        )
    flow = query_flow_manifests(root_uri=root_uri, status=FlowStatus.COMPLETED)
    assert len(flow) == 1
    assert len(flow[0].metadata["output_write_ids"]) == 4


# --------------------------------------------------------------------------- #
# 4. Silver flow
# --------------------------------------------------------------------------- #


def test_silver_flow_round_trips(root_uri: str) -> None:
    _seed_bronze(root_uri)

    industrial_load_silver(geographies=_geographies(), root_uri=root_uri)

    # one silver table per kind
    md = silver.read_metadata_silver(_industrial(), silver_root=root_uri)
    ts = silver.read_timeseries_silver(_industrial(), silver_root=root_uri)
    assert md.columns == silver.DsgridIndustrialMetadataSilverSchema.columns
    assert ts.columns == silver.DsgridIndustrialTimeseriesSilverSchema.columns
    assert ts["timestamp"].n_unique() == _N_HOURS
    assert set(ts[silver.SOURCE_COLUMN].unique()) == {str(INDUSTRIAL), str(GAPS)}

    flow = query_flow_manifests(root_uri=root_uri, status=FlowStatus.COMPLETED)
    assert len(flow) == 1
    assert len(flow[0].metadata["output_write_ids"]) == 2


def test_silver_flow_records_lineage(root_uri: str) -> None:
    """
    Each silver table joins a bronze *pair*, so with both sources requested the
    lineage names four datasets: metadata + timeseries per source.
    """
    _seed_bronze(root_uri)

    industrial_load_silver(geographies=_geographies(), root_uri=root_uri)

    flow = query_flow_manifests(root_uri=root_uri, status=FlowStatus.COMPLETED)[0]
    assert len(flow.input_ids) == 4

    lineage = read_flow_manifest(flow_id=flow.flow_id, root_uri=root_uri)
    assert lineage is not None
    # Keyed by state first: a run covering several states resolves a pair per
    # source per state, and "industrial_metadata" alone would not say which.
    assert set(lineage.metadata["resolved_inputs"]) == {_STATE}
    assert set(lineage.metadata["resolved_inputs"][_STATE]) == {
        f"{INDUSTRIAL}_metadata",
        f"{INDUSTRIAL}_timeseries",
        f"{GAPS}_metadata",
        f"{GAPS}_timeseries",
    }


def test_silver_flow_raises_when_bronze_missing(root_uri: str) -> None:
    with pytest.raises(RuntimeError, match="ingest the bronze slice"):
        industrial_load_silver(geographies=_geographies(), root_uri=root_uri)

    failed = query_flow_manifests(root_uri=root_uri, status=FlowStatus.FAILED)
    assert len(failed) == 1
    assert failed[0].flow_name == "industrial_load_silver"
    assert failed[0].error
    assert failed[0].end_time is not None


def test_silver_flow_honours_a_source_subset(root_uri: str) -> None:
    """
    A gaps-only request still writes both kinds of table -- narrowing ``sources``
    changes which rows are stacked, not how many tables there are -- and resolves
    only that source's bronze pair.
    """
    _seed_bronze(root_uri)
    industrial_load_silver(
        geographies=_geographies(),
        dsgrid=schema.DsgridRequestArgs(sources=(GAPS,)),
        root_uri=root_uri,
    )

    flow = query_flow_manifests(root_uri=root_uri, status=FlowStatus.COMPLETED)[0]
    assert set(flow.metadata["resolved_inputs"][_STATE]) == {
        f"{GAPS}_metadata",
        f"{GAPS}_timeseries",
    }
    assert len(flow.metadata["output_write_ids"]) == 2  # metadata + timeseries
    ts = silver.read_timeseries_silver(
        schema.IndustrialLoadRequestArgs(
            state=_STATE, dsgrid=schema.DsgridRequestArgs(sources=(GAPS,))
        ),
        silver_root=root_uri,
    )
    assert set(ts[silver.SOURCE_COLUMN].unique()) == {str(GAPS)}


def test_silver_flow_fails_loud_for_an_uningested_state(root_uri: str) -> None:
    """A state whose dsgrid bronze was never ingested must fail, not build a partial."""
    _seed_bronze(root_uri)
    with pytest.raises(RuntimeError, match="ingest the bronze slice"):
        industrial_load_silver(geographies=_geographies("G10000101"), root_uri=root_uri)

    failed = query_flow_manifests(root_uri=root_uri, status=FlowStatus.FAILED)
    assert len(failed) == 1


# --------------------------------------------------------------------------- #
# 5. Full pipeline
# --------------------------------------------------------------------------- #


def test_pipeline_runs_all_four(root_uri: str, offline_all: None) -> None:
    run_load_pipeline(geographies=_geographies(), root_uri=root_uri)

    # every bronze dataset landed under the single root_uri...
    for dataset_name in (
        resstock_bronze.METADATA_DATASET_NAME,
        resstock_bronze.TIMESERIES_DATASET_NAME,
        comstock_bronze.PUMA_METADATA_DATASET_NAME,
        comstock_bronze.TIMESERIES_DATASET_NAME,
    ):
        assert query_manifest(dataset_name=dataset_name, root_uri=root_uri)
    for source in dsgrid_bronze.PROFILES:
        assert query_manifest(
            dataset_name=dsgrid_bronze.PROFILES[source].timeseries_dataset_name,
            root_uri=root_uri,
        )
    # ...and so did both silver tables.
    for name in (silver.METADATA_DATASET_NAME, silver.TIMESERIES_DATASET_NAME):
        assert len(query_manifest(dataset_name=name, root_uri=root_uri)) == 1
    ts = silver.read_timeseries_silver(_industrial(), silver_root=root_uri)
    assert ts["timestamp"].n_unique() == _N_HOURS

    # Two flow manifests, not five: the three bronze fetches are tasks on this
    # flow's runner so they can overlap, and Prefect runs subflows sequentially.
    completed = query_flow_manifests(root_uri=root_uri, status=FlowStatus.COMPLETED)
    names = {flow.flow_name for flow in completed}
    assert names == {"run_load_pipeline", "industrial_load_silver"}
    # So the parent has to name the bronze writes itself -- it recorded nothing
    # before, leaving the per-source subflows as the only record of a run's output.
    parent = next(f for f in completed if f.flow_name == "run_load_pipeline")
    assert len(parent.metadata["output_write_ids"]) >= 6


def test_pipeline_marks_failed_and_reraises(
    root_uri: str, offline_all: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure inside a subflow must surface as a FAILED parent, not a silent pass."""

    def _boom(*args, **kwargs):
        msg = "OEDI exploded"
        raise RuntimeError(msg)

    monkeypatch.setattr(dsgrid_bronze, "ingest_all", _boom)

    with pytest.raises(RuntimeError, match="OEDI exploded"):
        run_load_pipeline(geographies=_geographies(), root_uri=root_uri)

    # One manifest now, since the bronze fetches are tasks rather than subflows.
    # ``.result()`` re-raises, so one failed source still fails the whole run --
    # per-source tolerance is a separate change, not a side effect of overlapping.
    failed = query_flow_manifests(root_uri=root_uri, status=FlowStatus.FAILED)
    names = {flow.flow_name for flow in failed}
    assert names == {"run_load_pipeline"}
    parent = next(flow for flow in failed if flow.flow_name == "run_load_pipeline")
    assert parent.error is not None
    assert "OEDI exploded" in parent.error
    assert parent.end_time is not None


# --------------------------------------------------------------------------- #
# A run of several PUMAs
# --------------------------------------------------------------------------- #


def test_resstock_writes_a_key_per_puma_off_one_state_download(
    root_uri: str, offline_two_pumas: list[str]
) -> None:
    """
    Two PUMAs, two metadata keys and two timeseries keys -- on one download.

    Both halves are addressed by the PUMA now, like every other dataset in the
    pipeline and like the climate pipeline's per-point writes. The file OEDI
    publishes is still per state, so the fetch is hoisted above the PUMA loop:
    a per-PUMA key must not turn into a download per PUMA.
    """
    ingest_resstock(geographies=_geographies(_PUMA, _SECOND_PUMA), root_uri=root_uri)

    for dataset_name in (
        resstock_bronze.METADATA_DATASET_NAME,
        resstock_bronze.TIMESERIES_DATASET_NAME,
    ):
        assert _pumas_written(dataset_name, root_uri) == {_PUMA, _SECOND_PUMA}
    assert len(_all_writes(resstock_bronze.METADATA_DATASET_NAME, root_uri)) == 2
    assert len(offline_two_pumas) == 1, offline_two_pumas

    # and each metadata write holds only the buildings its key claims
    for puma in (_PUMA, _SECOND_PUMA):
        table = resstock_bronze.read_metadata_bronze(
            silver.resstock_metadata_request(schema.LoadGeography(puma_gisjoin=puma)),
            root_uri,
        )
        assert table["puma_gisjoin"].unique().to_list() == [puma]

    # and each PUMA's write holds its own buildings, not the other's
    for puma in (_PUMA, _SECOND_PUMA):
        table = resstock_bronze.read_puma_timeseries_bronze(
            silver.resstock_timeseries_request(schema.LoadGeography(puma_gisjoin=puma)),
            root_uri,
        )
        assert set(table["puma_gisjoin"].unique()) == {puma}
        assert table["bldg_id"].n_unique() == 3


def test_comstock_ingests_every_puma(root_uri: str, offline_two_pumas: None) -> None:
    """ComStock's metadata is per PUMA, so both halves are keyed per PUMA."""
    ingest_comstock(geographies=_geographies(_PUMA, _SECOND_PUMA), root_uri=root_uri)

    for dataset_name in (
        comstock_bronze.PUMA_METADATA_DATASET_NAME,
        comstock_bronze.TIMESERIES_DATASET_NAME,
    ):
        assert _pumas_written(dataset_name, root_uri) == {_PUMA, _SECOND_PUMA}


def test_dsgrid_is_reconstructed_once_per_state_not_once_per_puma(
    root_uri: str, offline_dsgrid: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    dsgrid covers every county in a state, so a run's PUMAs collapse to its states.

    Reuse would make a second pass cheap, but not free: it would still submit a
    task and scan the ledger to learn what the first pass already established.
    """
    reconstructed: list[str] = []
    stubbed = dsgrid_bronze.reconstruct_tables

    def _counted(data, state, source=INDUSTRIAL):
        reconstructed.append(state)
        return stubbed(data, state, source)

    monkeypatch.setattr(dsgrid_bronze, "reconstruct_tables", _counted)

    ingest_dsgrid(geographies=_geographies(_PUMA, _SECOND_PUMA), root_uri=root_uri)

    # one reconstruction per source file, not per source file per PUMA
    assert reconstructed == [_STATE, _STATE]
    for profile in dsgrid_bronze.PROFILES.values():
        assert len(_all_writes(profile.timeseries_dataset_name, root_uri)) == 1


def test_silver_builds_one_table_per_state_not_one_per_puma(root_uri: str) -> None:
    """
    The silver tables are keyed by state, so two PUMAs of one state are one table
    per kind -- writing a second identical version per extra PUMA would say the
    run produced something it did not.
    """
    _seed_bronze(root_uri)

    industrial_load_silver(
        geographies=_geographies(_PUMA, _SECOND_PUMA), root_uri=root_uri
    )

    for dataset_name in (silver.METADATA_DATASET_NAME, silver.TIMESERIES_DATASET_NAME):
        assert len(_all_writes(dataset_name, root_uri)) == 1
    flow = query_flow_manifests(root_uri=root_uri, status=FlowStatus.COMPLETED)[0]
    assert len(flow.metadata["output_write_ids"]) == 2


def test_silver_resolves_every_state_before_it_builds_any(root_uri: str) -> None:
    """
    A run covering a state whose bronze is missing writes nothing at all.

    Resolving every state up front is what keeps a partly-satisfied request from
    leaving one state's silver behind and failing on the next -- the operator would
    have a table for a run that failed, and no signal that it is half a request.
    """
    _seed_bronze(root_uri)  # DC only

    with pytest.raises(RuntimeError, match="ingest the bronze slice"):
        industrial_load_silver(
            geographies=_geographies(_PUMA, "G10000101"),  # DC + DE
            root_uri=root_uri,
        )

    assert not query_manifest(
        dataset_name=silver.TIMESERIES_DATASET_NAME, root_uri=root_uri
    )
    assert len(query_flow_manifests(root_uri=root_uri, status=FlowStatus.FAILED)) == 1


def test_the_pipeline_covers_every_puma_and_collapses_the_state_work(
    root_uri: str, offline_two_pumas: list[str], offline_dsgrid: None
) -> None:
    """
    End to end for a list: PUMA-grained bronze per PUMA, state-grained work once.
    """
    run_load_pipeline(geographies=_geographies(_PUMA, _SECOND_PUMA), root_uri=root_uri)

    for dataset_name in (
        resstock_bronze.TIMESERIES_DATASET_NAME,
        comstock_bronze.PUMA_METADATA_DATASET_NAME,
        comstock_bronze.TIMESERIES_DATASET_NAME,
    ):
        assert _pumas_written(dataset_name, root_uri) == {_PUMA, _SECOND_PUMA}
    assert _pumas_written(resstock_bronze.METADATA_DATASET_NAME, root_uri) == {
        _PUMA,
        _SECOND_PUMA,
    }
    for dataset_name in (silver.METADATA_DATASET_NAME, silver.TIMESERIES_DATASET_NAME):
        assert len(_all_writes(dataset_name, root_uri)) == 1

    parent = next(
        flow
        for flow in query_flow_manifests(root_uri=root_uri, status=FlowStatus.COMPLETED)
        if flow.flow_name == "run_load_pipeline"
    )
    assert [geo["puma_gisjoin"] for geo in parent.metadata["geographies"]] == [
        _PUMA,
        _SECOND_PUMA,
    ]


# --------------------------------------------------------------------------- #
# Per-flow request schemas (what a deployment's run form exposes)
# --------------------------------------------------------------------------- #


def test_each_flow_exposes_only_the_schemas_it_uses(
    subtests: pytest.Subtests,
) -> None:
    """
    A deployment's run form is built from its flow signature, so each flow must
    take the combined geography plus exactly the source config(s) it touches.
    """
    expected = {
        ingest_resstock: {"geographies", "resstock"},
        ingest_comstock: {"geographies", "comstock"},
        ingest_dsgrid: {"geographies", "dsgrid"},
        industrial_load_silver: {"geographies", "dsgrid"},
        run_load_pipeline: {"geographies", "resstock", "comstock", "dsgrid"},
    }
    plumbing = {"root_uri", "writer", "as_of", "force_refresh"}
    for flow, schemas in expected.items():
        with subtests.test(flow.name):
            params = set(inspect.signature(flow.fn).parameters)
            assert params - plumbing == schemas


def test_dsgrid_flow_honors_a_single_source(
    root_uri: str, offline_dsgrid: None
) -> None:
    """Narrowing `sources` must ingest that half only."""
    ingest_dsgrid(
        geographies=_geographies(),
        dsgrid=schema.DsgridRequestArgs(sources=(GAPS,)),
        root_uri=root_uri,
    )
    gaps = dsgrid_bronze.PROFILES[GAPS]
    manufacturing = dsgrid_bronze.PROFILES[INDUSTRIAL]
    assert query_manifest(dataset_name=gaps.timeseries_dataset_name, root_uri=root_uri)
    assert not query_manifest(
        dataset_name=manufacturing.timeseries_dataset_name, root_uri=root_uri
    )


def test_building_stock_flows_thread_the_upgrade_into_the_bronze_key(
    root_uri: str, offline_resstock: None
) -> None:
    """
    `upgrade` now lives on the source config, so it must reach the bronze manifest
    key -- otherwise a non-baseline ingest would collide with the baseline.
    """
    ingest_resstock(
        geographies=_geographies(),
        resstock=schema.ResstockRequestArgs(upgrade=2),
        root_uri=root_uri,
    )
    rows = query_manifest(
        dataset_name=resstock_bronze.TIMESERIES_DATASET_NAME, root_uri=root_uri
    )
    assert len(rows) == 1
    assert '"upgrade":2' in rows[0].params_json.replace(" ", "")


def test_dsgrid_flow_threads_source_dataset_into_the_bronze_key(
    root_uri: str, offline_dsgrid: None
) -> None:
    """
    The dsgrid analogue of the ``upgrade`` test above.

    ``ingest_dsgrid`` used to let ``ingest_all`` rebuild the bronze key from
    state + source alone, which took the *default* ``source_dataset``. Silver
    derives the same key with the request's value, so an override wrote under
    one key and resolved under another: an ingest that succeeded, followed by a
    silver build reporting the bronze slice missing.
    """
    dsgrid = schema.DsgridRequestArgs(source_dataset="dsgrid-2018-efs-rev2")
    ingest_dsgrid(geographies=_geographies(), dsgrid=dsgrid, root_uri=root_uri)

    for profile in dsgrid_bronze.PROFILES.values():
        for dataset_name in (
            profile.metadata_dataset_name,
            profile.timeseries_dataset_name,
        ):
            rows = query_manifest(dataset_name=dataset_name, root_uri=root_uri)
            assert len(rows) == 1
            params = rows[0].params_json.replace(" ", "")
            assert '"source_dataset":"dsgrid-2018-efs-rev2"' in params

    # The written keys are exactly the ones silver resolves with -- the whole
    # point, since resolve_manifest matches params_json exactly.
    for request in silver.dsgrid_requests(_geography().state, dsgrid):
        profile = dsgrid_bronze.PROFILES[request.source]
        assert resolve_manifest(
            dataset_name=profile.timeseries_dataset_name,
            params=request,
            root_uri=root_uri,
        )


def test_pipeline_with_a_past_as_of_still_builds_silver(
    root_uri: str, offline_all: None
) -> None:
    """
    ``run_load_pipeline`` must not thread ``as_of`` into the silver step.

    Steps 1-3 write bronze at ``now`` and silver resolves ``write_time <=
    as_of``, so threading a past ``as_of`` through would exclude the run's own
    bronze and fail at step 4 -- after paying for the building-stock fetches.
    """
    run_load_pipeline(
        geographies=_geographies(),
        root_uri=root_uri,
        as_of=datetime.datetime.now(tz=datetime.UTC) - datetime.timedelta(days=1),
    )
    assert query_manifest(
        dataset_name=silver.TIMESERIES_DATASET_NAME, root_uri=root_uri
    )


# --------------------------------------------------------------------------- #
# Timeout arithmetic
# --------------------------------------------------------------------------- #


def test_the_pipeline_timeout_reflects_concurrent_bronze() -> None:
    """
    The parent's budget is the slowest bronze source plus silver, not their sum.

    The sum was right while the three ran as sequential subflows and became wrong
    the moment they were overlapped. Before that it was *below* the sum -- 28800 s
    against 32400 s -- so the parent could cancel a run whose children were each
    still inside their own budget. Deriving it either way is what stops both
    mistakes, so this asserts the derivation rather than a number.
    """
    units = flows.MAX_UNITS_PER_RUN
    sources = (flows.BUILDING_STOCK_TIMEOUT_SECONDS, flows.DSGRID_TIMEOUT_SECONDS)
    assert run_load_pipeline.timeout_seconds == (
        flows.METADATA_PHASE_TIMEOUT_SECONDS
        + units * (max(sources) + flows.SILVER_TIMEOUT_SECONDS)
    )
    # overlapping is what makes the parent smaller than running them in turn
    sequential = flows.METADATA_PHASE_TIMEOUT_SECONDS + units * (
        2 * flows.BUILDING_STOCK_TIMEOUT_SECONDS
        + flows.DSGRID_TIMEOUT_SECONDS
        + flows.SILVER_TIMEOUT_SECONDS
    )
    assert run_load_pipeline.timeout_seconds < sequential

    # ...and no phase is missing from the sum. The metadata phase was: it runs
    # before the PUMA loop, one state at a time, and neither side of this
    # assertion named it -- so the parent could sit below the sum of its children
    # and the test would still pass, which is the failure it exists to catch.
    worst_case_children = flows.METADATA_PHASE_TIMEOUT_SECONDS + units * (
        max(sources) + flows.SILVER_TIMEOUT_SECONDS
    )
    assert run_load_pipeline.timeout_seconds >= worst_case_children
    assert ingest_resstock.timeout_seconds >= (
        flows.METADATA_PHASE_TIMEOUT_SECONDS
        + units * flows.BUILDING_STOCK_TIMEOUT_SECONDS
    )
    # every standalone flow still carries one of the derived constants, widened by
    # the same factor -- a run is up to that many PUMAs, walked one at a time
    assert {
        ingest_resstock.timeout_seconds,
        ingest_comstock.timeout_seconds,
        ingest_dsgrid.timeout_seconds,
        industrial_load_silver.timeout_seconds,
    } == {
        flows.BUILDING_STOCK_FLOW_TIMEOUT_SECONDS,
        units * flows.DSGRID_TIMEOUT_SECONDS,
        units * flows.SILVER_TIMEOUT_SECONDS,
    }


def test_one_units_budget_is_enforced_on_the_task_that_does_it() -> None:
    """
    The hang-catcher sits on the **task**, which is one PUMA (or one state).

    A flow budget wide enough for a full list of PUMAs cannot also be tight around
    one of them: a single stuck fetch would sit for the length of the whole run
    before anything fired. So each unit of work carries the per-unit budget, and
    the flow's is only the ceiling on the run as a whole.
    """
    assert flows.fetch_resstock_bronze.timeout_seconds == (
        flows.BUILDING_STOCK_TIMEOUT_SECONDS
    )
    assert flows.fetch_comstock_bronze.timeout_seconds == (
        flows.BUILDING_STOCK_TIMEOUT_SECONDS
    )
    assert flows.fetch_dsgrid_bronze.timeout_seconds == flows.DSGRID_TIMEOUT_SECONDS
    assert flows.build_industrial_silver_table.timeout_seconds == (
        flows.SILVER_TIMEOUT_SECONDS
    )
    # A metadata step is one download, not an hour of per-building fetching
    assert flows.fetch_resstock_metadata_bronze.timeout_seconds == (
        flows.METADATA_TIMEOUT_SECONDS
    )
    assert flows.fetch_dsgrid_source_bronze.timeout_seconds == (
        flows.DSGRID_TIMEOUT_SECONDS
    )
    assert flows.fetch_comstock_metadata_bronze.timeout_seconds == (
        flows.METADATA_TIMEOUT_SECONDS
    )
    # and every flow leaves room for the units it schedules
    for flow in (ingest_resstock, ingest_comstock, ingest_dsgrid):
        budget = flow.timeout_seconds
        assert budget is not None
        assert budget >= flows.MAX_UNITS_PER_RUN * flows.DSGRID_TIMEOUT_SECONDS


def test_the_pipeline_overlaps_the_three_sources(
    root_uri: str, offline_all: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    All three bronze fetches must be in flight at once.

    Proven by making each source wait to see the other two: run them one after
    another and no wait can succeed, because the first has finished before the
    second starts. This is the whole reason they are tasks on the pipeline's task
    runner rather than subflows -- Prefect runs subflows sequentially however the
    work inside them is written.

    Both building-stock sources are seamed at the metadata *read* rather than the
    parse. Each parse happens in its own phase above the PUMA loop, which overlaps
    within itself (see ``test_the_metadata_fetches_overlap``) but not with this --
    so neither parse is evidence either way here. The read is the first thing each
    PUMA's timeseries task does, which is the work being overlapped.
    """
    in_flight = {n: threading.Event() for n in ("resstock", "comstock", "dsgrid")}
    saw_the_others: dict[str, bool] = {}

    def _wrap(name: str, original):
        def _seam(*args, **kwargs):
            if name not in saw_the_others:
                in_flight[name].set()
                others = [e for k, e in in_flight.items() if k != name]
                saw_the_others[name] = all(e.wait(timeout=30) for e in others)
            return original(*args, **kwargs)

        return _seam

    for module, attr, name in (
        (resstock_bronze, "read_metadata_bronze", "resstock"),
        (comstock_bronze, "read_puma_metadata_bronze", "comstock"),
        (dsgrid_bronze, "reconstruct_tables", "dsgrid"),
    ):
        monkeypatch.setattr(module, attr, _wrap(name, getattr(module, attr)))

    run_load_pipeline(geographies=_geographies(), root_uri=root_uri)

    assert saw_the_others == {"resstock": True, "comstock": True, "dsgrid": True}


def test_the_per_building_fetches_actually_overlap(
    root_uri: str, offline_all: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The fan-out must really fan out.

    It did not. With ``max_workers=3`` the three parent fetch tasks held every
    worker in the pool while they blocked on the children they had mapped, and
    those children draw from the same pool -- so run 3b859d8e fetched 115
    buildings strictly one at a time, and the first one did not start until dsgrid
    finished and released a slot 21 s in. Nothing was broken, everything was
    ordered, and it took as long as the sum of its parts.

    Proven the way the source overlap is: a fetch that waits to see another fetch
    in flight can only succeed if two are running at once.
    """
    seen_together = threading.Event()
    in_flight = threading.Semaphore(0)
    original = resstock_bronze.fetch_building_frame

    def _seam(args, bldg_id, client):
        in_flight.release()
        # The first fetch to arrive waits for a second; whoever satisfies it
        # proves the two overlapped. Bounded so a serial runner fails the assert
        # rather than hanging the suite.
        if in_flight.acquire(timeout=20) and in_flight.acquire(timeout=20):
            seen_together.set()
            in_flight.release()
        in_flight.release()
        return original(args, bldg_id, client)

    monkeypatch.setattr(resstock_bronze, "fetch_building_frame", _seam)

    run_load_pipeline(geographies=_geographies(), root_uri=root_uri)

    assert seen_together.is_set(), (
        "no two building fetches were ever in flight together -- the task pool is "
        "starving the mapped children again"
    )


def test_the_concurrency_knobs_come_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Both knobs are tunable per environment, and the pool follows them.

    They are tuned to whatever host the flows are imported on: this machine is
    bandwidth-bound at ~28 in flight, a VM in the bucket's own region is not. The
    defaults are the measured local ceiling, so production must be able to raise
    them **without a code change** -- and when it does, ``TASK_WORKERS`` has to
    move too, or the pool silently caps the fan-out below the HTTP budget. That
    already happened once: ``oedi-api`` was raised to 48 server-side and the
    fetches stayed pinned at 28.

    Re-imported under a patched environment because the values feed a flow
    decorator, so they are read once at import rather than per run.
    """
    monkeypatch.setenv("LOAD_PIPELINE_PUMAS_IN_FLIGHT", "4")
    monkeypatch.setenv("LOAD_PIPELINE_OEDI_IN_FLIGHT", "64")
    reloaded = importlib.reload(flows)
    try:
        assert reloaded.PUMAS_IN_FLIGHT == 4
        assert reloaded.OEDI_IN_FLIGHT == 64
        # the pool still leaves the whole HTTP budget free with every parent parked
        parents = reloaded.PUMAS_IN_FLIGHT * (
            reloaded.SOURCES_PER_PUMA + reloaded.CHILD_TASKS_PER_PUMA
        )
        assert reloaded.TASK_WORKERS - parents >= reloaded.OEDI_IN_FLIGHT
        assert reloaded.run_load_pipeline.task_runner is not None
        assert (
            reloaded.run_load_pipeline.task_runner._max_workers == reloaded.TASK_WORKERS
        )
    finally:
        # restore the module other tests hold references to
        monkeypatch.delenv("LOAD_PIPELINE_PUMAS_IN_FLIGHT")
        monkeypatch.delenv("LOAD_PIPELINE_OEDI_IN_FLIGHT")
        importlib.reload(flows)


def test_a_mistyped_concurrency_knob_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A typo must not fall back to the default.

    Silently using 1 where the operator asked for 8 gives a run that looks healthy
    and is sized for the wrong machine -- the kind of thing found weeks later in a
    throughput graph, if at all. Nonsense and out-of-range are both refused; an
    unset variable is the only thing that takes the default.
    """
    monkeypatch.setenv("LOAD_PIPELINE_TEST_KNOB", "four")
    with pytest.raises(RuntimeError, match="must be an integer"):
        flows._env_int("LOAD_PIPELINE_TEST_KNOB", 1)

    monkeypatch.setenv("LOAD_PIPELINE_TEST_KNOB", "0")
    with pytest.raises(RuntimeError, match="must be >= 1"):
        flows._env_int("LOAD_PIPELINE_TEST_KNOB", 1)

    monkeypatch.delenv("LOAD_PIPELINE_TEST_KNOB")
    assert flows._env_int("LOAD_PIPELINE_TEST_KNOB", 7) == 7


def test_the_pool_leaves_room_for_the_children_it_maps() -> None:
    """
    The pool has to hold every parent *and* the HTTP budget.

    This is the arithmetic that was missing: sizing it to the parents alone is what
    let three tasks occupy every worker and serialise the fetches they mapped. A
    pool at or below the parent count is the bug, restated.
    """
    # Parents *and* the children they map: the dsgrid parent maps one task per
    # source file, so splitting that task for visibility spends workers too. The
    # metadata fetches are flow-level phases, so no parent waits on them.
    parents = flows.PUMAS_IN_FLIGHT * (
        flows.SOURCES_PER_PUMA + flows.CHILD_TASKS_PER_PUMA
    )
    # The pool must not shrink with the PUMA window: the per-building fan-out is
    # bounded by the HTTP budget, not by how many PUMAs are open.
    assert flows.TASK_WORKERS >= parents + flows.OEDI_IN_FLIGHT
    for flow in (ingest_resstock, ingest_comstock, ingest_dsgrid, run_load_pipeline):
        runner = flow.task_runner
        assert runner is not None, flow.name
        assert runner._max_workers >= parents + flows.OEDI_IN_FLIGHT, flow.name


def test_pumas_are_fetched_one_at_a_time(
    root_uri: str, offline_two_pumas: list[str], offline_dsgrid: None
) -> None:
    """
    A run opens one PUMA at a time, and that is a memory bound, not a courtesy.

    PUMAs used to overlap four at a time, which was right while each was written in
    50-building chunks and nothing large was ever resident. A PUMA is one write now:
    every building's frame is held until the whole table can be concatenated, which
    measures 1.2 GB for a full ComStock PUMA and a process high-water mark several
    times that. Two PUMAs open would double it, and the two sources of a single
    PUMA already overlap.

    The concurrency that matters is untouched -- the per-building fan-out inside a
    PUMA still runs at the HTTP budget's ceiling, which is what
    ``test_the_per_building_fetches_actually_overlap`` measures. So raising
    ``PUMAS_IN_FLIGHT`` means measuring the machine's memory, and this fails until
    someone does.
    """
    assert flows.PUMAS_IN_FLIGHT == 1

    in_flight: set[str] = set()
    peak = 0
    lock = threading.Lock()
    original = resstock_bronze.fetch_building_frame

    def _seam(args, bldg_id, client):
        nonlocal peak
        with lock:
            in_flight.add(args.puma_gisjoin)
            peak = max(peak, len(in_flight))
        try:
            return original(args, bldg_id, client)
        finally:
            with lock:
                in_flight.discard(args.puma_gisjoin)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(resstock_bronze, "fetch_building_frame", _seam)
    try:
        run_load_pipeline(
            geographies=_geographies(_PUMA, _SECOND_PUMA), root_uri=root_uri
        )
    finally:
        monkeypatch.undo()

    assert peak == 1, f"{peak} PUMAs had fetches in flight together"
    # both PUMAs still landed -- serialised, not skipped
    assert _pumas_written(resstock_bronze.TIMESERIES_DATASET_NAME, root_uri) == {
        _PUMA,
        _SECOND_PUMA,
    }


def test_comstock_metadata_is_its_own_unit_of_work(
    root_uri: str, offline_comstock: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    ComStock's metadata is a **phase of the flow**, like ResStock's -- not a task
    submitted from inside the bronze task.

    It was the only part of a ComStock ingest with no row of its own, so a run read
    as "886 building fetches and nothing else". Making it a task fixed that;
    hoisting it to the flow fixed two more things. A bronze task no longer holds a
    runner worker while blocking on a child of its own, and the write ordering the
    weights depend on is now structural -- the whole metadata phase completes
    before any timeseries task starts, rather than resting on where a
    ``write_time`` is stamped inside one task.
    """
    seen: list[str] = []
    original = comstock_bronze.ingest_puma_metadata_bronze

    def _spy(params, **kwargs):
        seen.append(params.puma_gisjoin)
        return original(params, **kwargs)

    monkeypatch.setattr(comstock_bronze, "ingest_puma_metadata_bronze", _spy)

    ingest_comstock(geographies=_geographies(), root_uri=root_uri)

    assert seen == [_PUMA]
    metadata = query_manifest(
        dataset_name=comstock_bronze.PUMA_METADATA_DATASET_NAME, root_uri=root_uri
    )
    timeseries = query_manifest(
        dataset_name=comstock_bronze.TIMESERIES_DATASET_NAME, root_uri=root_uri
    )
    assert len(metadata) == 1 and len(timeseries) == 1
    assert metadata[0].write_time <= timeseries[0].write_time

    # every metadata write lands before any timeseries write, across all PUMAs --
    # which is what "a phase" buys over "a task inside a task"
    all_metadata = _all_writes(comstock_bronze.PUMA_METADATA_DATASET_NAME, root_uri)
    all_timeseries = _all_writes(comstock_bronze.TIMESERIES_DATASET_NAME, root_uri)
    assert max(w.write_time for w in all_metadata) <= min(
        w.write_time for w in all_timeseries
    )


def test_comstock_metadata_is_a_phase_before_any_timeseries(
    root_uri: str, offline_two_pumas: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Across several PUMAs, every metadata write precedes every timeseries fetch.

    That is the difference between a phase and a nested task: submitted from
    inside ``fetch_comstock_bronze``, PUMA 2's metadata was fetched *after* PUMA
    1's profiles. Now the phase drains first, so the ordering holds run-wide rather
    than only within a PUMA -- and no bronze task sits on a worker waiting for a
    child it submitted.
    """
    order: list[str] = []
    original_meta = comstock_bronze.ingest_puma_metadata_bronze
    original_fetch = comstock_bronze.fetch_building_frame

    def _meta(params, **kwargs):
        order.append(f"metadata:{params.puma_gisjoin}")
        return original_meta(params, **kwargs)

    def _fetch(args, bldg_id, client):
        tag = f"timeseries:{args.puma_gisjoin}"
        if tag not in order:
            order.append(tag)
        return original_fetch(args, bldg_id, client)

    monkeypatch.setattr(comstock_bronze, "ingest_puma_metadata_bronze", _meta)
    monkeypatch.setattr(comstock_bronze, "fetch_building_frame", _fetch)

    ingest_comstock(geographies=_geographies(_PUMA, _SECOND_PUMA), root_uri=root_uri)

    kinds = [entry.split(":", 1)[0] for entry in order]
    assert kinds == ["metadata", "metadata", "timeseries", "timeseries"], order


def test_comstock_metadata_is_reused_without_a_refetch(
    root_uri: str, offline_comstock: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reuse decision moved into the task with the work, so it still holds."""
    ingest_comstock(geographies=_geographies(), root_uri=root_uri)
    monkeypatch.setattr(comstock_bronze, "parquet_to_puma_metadata_table", _explode)

    ingest_comstock(geographies=_geographies(), root_uri=root_uri)

    assert len(_all_writes(comstock_bronze.PUMA_METADATA_DATASET_NAME, root_uri)) == 1


def test_each_dsgrid_source_is_its_own_unit_of_work(
    root_uri: str, offline_dsgrid: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    One task run per ``.dsg`` file, not one for both.

    A source is a download, an HDF5 reconstruction and two writes. Doing both
    inside one task made the most expensive step in the industrial path a box with
    no progress in it -- an operator could not tell a slow file from a stalled one,
    or which of the two was being paid for. Proven by what reaches the package: one
    call per source, each carrying exactly its own key.
    """
    calls: list[list[str]] = []
    original = dsgrid_bronze.ingest_all

    def _spy(*args, requests=None, **kwargs):
        calls.append([str(params.source) for params in requests or []])
        return original(*args, requests=requests, **kwargs)

    monkeypatch.setattr(dsgrid_bronze, "ingest_all", _spy)

    ingest_dsgrid(geographies=_geographies(), root_uri=root_uri)

    assert calls == [[str(INDUSTRIAL)], [str(GAPS)]] or calls == [
        [str(GAPS)],
        [str(INDUSTRIAL)],
    ], calls
    # ...and all four datasets still landed
    for profile in dsgrid_bronze.PROFILES.values():
        for name in (profile.metadata_dataset_name, profile.timeseries_dataset_name):
            assert query_manifest(dataset_name=name, root_uri=root_uri)


def test_a_reused_dsgrid_source_is_decided_per_source(
    root_uri: str, offline_dsgrid: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The reuse decision moved with the work, so it is still made per source file.

    A run that already holds one half must fetch only the other -- which is the
    whole point of the per-source key, and would be lost if the split had hoisted
    the ledger check back up to the state.
    """
    ingest_dsgrid(
        geographies=_geographies(),
        dsgrid=schema.DsgridRequestArgs(sources=(GAPS,)),
        root_uri=root_uri,
    )
    reconstructed: list[str] = []
    stubbed = dsgrid_bronze.reconstruct_tables

    def _counted(data, state, source=INDUSTRIAL):
        reconstructed.append(str(source))
        return stubbed(data, state, source)

    monkeypatch.setattr(dsgrid_bronze, "reconstruct_tables", _counted)

    ingest_dsgrid(geographies=_geographies(), root_uri=root_uri)

    assert reconstructed == [str(INDUSTRIAL)], reconstructed
    for profile in dsgrid_bronze.PROFILES.values():
        assert len(_all_writes(profile.timeseries_dataset_name, root_uri)) == 1


def test_each_silver_table_is_its_own_unit_of_work(
    root_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A task run per silver table, so the expensive one says so.

    The two tables read different bronze pairs and share nothing but the request,
    so building them together bought a line of code and cost any sight of which
    one a run was inside -- and they are not the same size: 237k timeseries rows
    for DC against 27 metadata rows.
    """
    _seed_bronze(root_uri)
    built: list[str] = []
    for name in ("build_metadata_table", "build_timeseries_table"):
        original = getattr(silver, name)

        def _spy(*args, _name=name, _original=original, **kwargs):
            built.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(silver, name, _spy)

    industrial_load_silver(geographies=_geographies(), root_uri=root_uri)

    assert sorted(built) == ["build_metadata_table", "build_timeseries_table"]
    for name in (silver.METADATA_DATASET_NAME, silver.TIMESERIES_DATASET_NAME):
        assert len(_all_writes(name, root_uri)) == 1


def test_an_unknown_silver_table_is_refused() -> None:
    """
    The kind crosses a task boundary as a string, so it is resolved rather than
    trusted -- and it fails before any bronze is read.
    """
    with pytest.raises(PipelineValueError, match="unknown silver table"):
        flows._silver_builder("gold")
    assert set(flows.SILVER_TABLES) == {"metadata", "timeseries"}


def test_the_measured_fan_out_fits_inside_a_building_stock_budget() -> None:
    """
    The budgets are sized from a measured fetch rate, so the estimate has to fit.

    ~22 requests/second measured at 8-way concurrency against the live bucket,
    rounded down to 20: a ComStock PUMA is ~48 s of fetching against a 3600 s
    budget. If the rate is ever revised to something a budget cannot absorb, this
    fails here rather than as a run cancelled mid-fetch.
    """
    assert flows.RESSTOCK_FETCH_SECONDS < flows.BUILDING_STOCK_TIMEOUT_SECONDS
    assert flows.COMSTOCK_FETCH_SECONDS < flows.BUILDING_STOCK_TIMEOUT_SECONDS


# --------------------------------------------------------------------------- #
# Reuse: the ledger is consulted before fetching
# --------------------------------------------------------------------------- #


def _all_writes(dataset_name: str, root_uri: str) -> list:
    """Every write, not the latest per params -- which is what reuse is about."""
    return query_manifest(
        dataset_name=dataset_name, root_uri=root_uri, latest_per_params=False
    )


def _explode(*args, **kwargs):
    msg = "re-fetched a slice the ledger already held"
    raise AssertionError(msg)


def test_resstock_bronze_flow_reuses_both_halves(
    root_uri: str, offline_resstock: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A second identical ingest must touch the network zero times.

    Asserted by making every fetch seam raise: if reuse worked, none of them is
    reached. Counting manifest rows alone would not prove it -- a re-fetch that
    wrote an identical second version would still leave one *latest* row.
    """
    ingest_resstock(geographies=_geographies(), root_uri=root_uri)
    for seam in (
        "download_object",
        "parquet_to_metadata_table",
        "resolve_puma_building_ids",
        "fetch_puma_timeseries_table",
        "fetch_building_frame",
    ):
        monkeypatch.setattr(resstock_bronze, seam, _explode)

    ingest_resstock(geographies=_geographies(), root_uri=root_uri)

    for name in (
        resstock_bronze.METADATA_DATASET_NAME,
        resstock_bronze.TIMESERIES_DATASET_NAME,
    ):
        assert len(_all_writes(name, root_uri)) == 1


def test_reused_writes_are_still_recorded_as_the_run_s_output(
    root_uri: str, offline_resstock: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A run that reused everything still has to say what it produced.

    Lineage asks "what did this run leave behind", not "what did it download", so
    a reusing run names the same write_ids as the run that fetched them. Recording
    an empty list would make a cheap run indistinguishable from a failed one.
    """
    ingest_resstock(geographies=_geographies(), root_uri=root_uri)
    first = query_flow_manifests(root_uri=root_uri, status=FlowStatus.COMPLETED)[0]
    for seam in (
        "download_object",
        "parquet_to_metadata_table",
        "resolve_puma_building_ids",
        "fetch_puma_timeseries_table",
        "fetch_building_frame",
    ):
        monkeypatch.setattr(resstock_bronze, seam, _explode)

    ingest_resstock(geographies=_geographies(), root_uri=root_uri)

    runs = query_flow_manifests(root_uri=root_uri, status=FlowStatus.COMPLETED)
    assert len(runs) == 2
    second = next(r for r in runs if r.flow_id != first.flow_id)
    assert set(second.metadata["output_write_ids"]) == set(
        first.metadata["output_write_ids"]
    )


def test_force_refresh_rewrites_a_covered_slice(
    root_uri: str, offline_resstock: None
) -> None:
    """``force_refresh`` is the escape hatch for a release re-published under an
    unchanged key, so it must fetch and write a second immutable version."""
    ingest_resstock(geographies=_geographies(), root_uri=root_uri)
    ingest_resstock(geographies=_geographies(), root_uri=root_uri, force_refresh=True)

    writes = _all_writes(resstock_bronze.TIMESERIES_DATASET_NAME, root_uri)
    assert len(writes) == 2
    # same key, two versions -- a reader still resolves the newer one
    assert len({w.params_json for w in writes}) == 1


def test_a_present_half_is_reused_while_the_missing_half_is_fetched(
    root_uri: str, offline_resstock: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The resumability case: a run that died between the two writes.

    Only the metadata is on disk, so the ingest must reuse it and fetch just the
    timeseries -- which is the half worth hundreds of requests.
    """
    columnar.write_dataset(
        _fixture("resstock_metadata_bronze"),
        resstock_bronze.ResstockMetadataBronzeSchema,
        resstock_bronze.METADATA_DATASET_NAME,
        silver.resstock_metadata_request(_geography()),
        root_uri,
        writer="seed",
    )
    # Reaching the metadata parse at all means the reuse check missed.
    monkeypatch.setattr(resstock_bronze, "parquet_to_metadata_table", _explode)

    ingest_resstock(geographies=_geographies(), root_uri=root_uri)

    assert len(_all_writes(resstock_bronze.METADATA_DATASET_NAME, root_uri)) == 1
    assert len(_all_writes(resstock_bronze.TIMESERIES_DATASET_NAME, root_uri)) == 1


def test_dsgrid_bronze_is_reused_across_pumas_in_one_state(
    root_uri: str, offline_dsgrid: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    dsgrid is keyed by *state*, so a second PUMA in the same state needs nothing.

    Before the ledger scan this re-downloaded both national ``.dsg`` files -- 15 MB
    and 2.6 MB of state-independent data -- once per PUMA. This is the cheapest
    reuse in the pipeline and the one that compounds fastest across a state.
    """
    ingest_dsgrid(geographies=_geographies(), root_uri=root_uri)
    monkeypatch.setattr(dsgrid_bronze.dsg_common, "load_dsg_bytes", _explode)
    monkeypatch.setattr(dsgrid_bronze, "reconstruct_tables", _explode)

    # a different PUMA, same state (both are DC)
    ingest_dsgrid(geographies=_geographies("G11000105"), root_uri=root_uri)

    for source in dsgrid_bronze.PROFILES:
        profile = dsgrid_bronze.PROFILES[source]
        for name in (profile.metadata_dataset_name, profile.timeseries_dataset_name):
            assert len(_all_writes(name, root_uri)) == 1


def test_comstock_reuses_both_datasets(
    root_uri: str, offline_comstock: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A re-run of an ingested PUMA touches the network zero times."""
    ingest_comstock(geographies=_geographies(), root_uri=root_uri)
    before = len(_all_writes(comstock_bronze.TIMESERIES_DATASET_NAME, root_uri))
    for seam in (
        "download_object",
        "parquet_to_puma_metadata_table",
        "resolve_puma_building_ids",
        "fetch_puma_timeseries_table",
        "fetch_building_frame",
    ):
        monkeypatch.setattr(comstock_bronze, seam, _explode)

    ingest_comstock(geographies=_geographies(), root_uri=root_uri)

    assert len(_all_writes(comstock_bronze.PUMA_METADATA_DATASET_NAME, root_uri)) == 1
    assert len(_all_writes(comstock_bronze.TIMESERIES_DATASET_NAME, root_uri)) == before


# --------------------------------------------------------------------------- #
# One write per PUMA: what its key claims, and what an interruption costs
# --------------------------------------------------------------------------- #


def test_a_puma_is_written_as_one_dataset(
    root_uri: str, offline_resstock: None
) -> None:
    """
    One manifest entry per PUMA, keyed by exactly what a reader can construct.

    It was one entry per 50-building chunk keyed by an id range, then one per PUMA
    carrying the building count it held. Both put something in the key that a
    caller asking for a PUMA could not know in advance, which is why reads had to
    scan rather than resolve. The key is now the request itself.
    """
    ingest_resstock(geographies=_geographies(), root_uri=root_uri)

    writes = _all_writes(resstock_bronze.TIMESERIES_DATASET_NAME, root_uri)
    assert len(writes) == 1, "expected the whole PUMA in one write"
    key = json.loads(writes[0].params_json)
    assert key["puma_gisjoin"] == _PUMA
    # nothing in the key that describes what the fetch returned
    for absent in ("n_buildings", "first_bldg_id", "last_bldg_id"):
        assert absent not in key, key
    # so the key a reader builds resolves the write directly
    assert key == json.loads(
        silver.resstock_timeseries_request(_geography()).model_dump_json()
    )

    # one parquet file, and every building in it
    assert (
        len(
            list(
                pathlib.Path(root_uri).rglob(
                    f"{resstock_bronze.TIMESERIES_DATASET_NAME}/*.parquet"
                )
            )
        )
        == 1
    )
    back = resstock_bronze.read_puma_timeseries_bronze(
        silver.resstock_timeseries_request(_geography()), root_uri
    )
    assert (
        back["bldg_id"].n_unique()
        == _fixture("resstock_timeseries_bronze")["bldg_id"].n_unique()
    )


def test_a_lost_write_re_fetches_the_whole_puma(
    root_uri: str,
    offline_resstock: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    What one file per PUMA costs: an interrupted PUMA resumes from nothing.

    Chunking used to bound this -- a re-run fetched only the chunks it was missing,
    so a killed run kept its work. With a single write there is no partial state to
    resume from: the write either exists and is reused whole, or it does not and
    every building is fetched again. Asserted rather than left implicit, because it
    is the trade the single file was chosen for.
    """
    ingest_resstock(geographies=_geographies(), root_uri=root_uri)
    writes = _all_writes(resstock_bronze.TIMESERIES_DATASET_NAME, root_uri)
    assert len(writes) == 1
    all_buildings = _fixture("resstock_timeseries_bronze")["bldg_id"].n_unique()

    sidecars = sorted(
        (pathlib.Path(root_uri) / "_manifests").rglob(f"*{writes[0].write_id}*")
    )
    assert sidecars, "expected a manifest sidecar for the write"
    sidecars[0].unlink()

    fetched: list[int] = []
    original = resstock_bronze.fetch_building_frame

    def _counting(args, bldg_id, client):
        fetched.append(bldg_id)
        return original(args, bldg_id, client)

    monkeypatch.setattr(resstock_bronze, "fetch_building_frame", _counting)
    ingest_resstock(geographies=_geographies(), root_uri=root_uri)

    assert len(fetched) == all_buildings, "a lost write re-fetches the PUMA whole"


def test_an_existing_write_is_reused_without_a_single_fetch(
    root_uri: str, offline_resstock: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other side of it: a PUMA already written costs nothing to re-run."""
    ingest_resstock(geographies=_geographies(), root_uri=root_uri)
    before = len(_all_writes(resstock_bronze.TIMESERIES_DATASET_NAME, root_uri))
    monkeypatch.setattr(resstock_bronze, "fetch_building_frame", _explode)

    ingest_resstock(geographies=_geographies(), root_uri=root_uri)

    assert len(_all_writes(resstock_bronze.TIMESERIES_DATASET_NAME, root_uri)) == before


def test_buildings_within_a_puma_are_fetched_concurrently(
    root_uri: str, offline_resstock: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The per-building fan-out has to actually fan out.

    A whole PUMA used to be one task holding a thread pool inside the domain
    package; it is now one mapped task per building. This measures the thing that
    change was for -- the peak number of fetches in flight at once -- rather than
    asserting a seam was reached, which a serial run would also satisfy.

    Each call holds its slot briefly so an overlap is observable at all; without
    the hold, three fast fetches could interleave without ever coinciding.
    """
    lock = threading.Lock()
    active = 0
    peak = 0
    original = resstock_bronze.fetch_building_frame

    def _seam(args, bldg_id, client):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.25)
            return original(args, bldg_id, client)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(resstock_bronze, "fetch_building_frame", _seam)
    ingest_resstock(geographies=_geographies(), root_uri=root_uri)

    assert peak > 1, f"per-building fetches ran one at a time (peak={peak})"


# --------------------------------------------------------------------------- #
# Partial failure: a PUMA completes over a building the release cannot serve
# --------------------------------------------------------------------------- #


def _doomed_seam(doomed: int, transient: bool = False):
    """A per-building seam that fails for one id and serves the rest."""
    original = resstock_bronze.fetch_building_frame

    def _seam(args, bldg_id, client):
        if bldg_id == doomed:
            if transient:
                msg = f"transient trouble for {bldg_id}"
                raise PipelineError(msg)
            msg = f"OEDI cannot serve this request (404) for building {bldg_id}"
            raise failures.PermanentFetchError(msg)
        return original(args, bldg_id, client)

    return _seam


def test_a_run_that_dies_partway_still_records_what_it_refused(
    root_uri: str, offline_two_pumas: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A failed run must keep the dead buildings it discovered before it failed.

    ``failed`` used to be returned by the walker and recorded only on the success
    path, so a run that died in a later PUMA threw away every permanently
    unavailable building the earlier PUMAs had found. The next run would ask for
    them again and could die the same way -- the re-asking loop ``known_dead``
    exists to break, moved up a level from the building to the run.

    Here the first PUMA has one building the release refuses, and the second PUMA
    fails outright. The refusal must survive on the FAILED manifest.
    """
    doomed = sorted(
        _fixture("resstock_timeseries_bronze")["bldg_id"].unique().to_list()
    )[0]
    original = resstock_bronze.fetch_building_frame

    def _seam(args, bldg_id, client):
        if args.puma_gisjoin == _SECOND_PUMA:
            msg = "the second PUMA falls over"
            raise PipelineError(msg)
        if bldg_id == doomed:
            msg = f"OEDI cannot serve this request (404) for building {bldg_id}"
            raise failures.PermanentFetchError(msg)
        return original(args, bldg_id, client)

    monkeypatch.setattr(resstock_bronze, "fetch_building_frame", _seam)
    monkeypatch.setattr(
        flows,
        "fetch_resstock_building",
        flows.fetch_resstock_building.with_options(retries=0),
    )

    with pytest.raises(Exception, match="falls over"):
        ingest_resstock(
            geographies=_geographies(_PUMA, _SECOND_PUMA), root_uri=root_uri
        )

    failed_runs = query_flow_manifests(root_uri=root_uri, status=FlowStatus.FAILED)
    assert len(failed_runs) == 1
    recorded = (failed_runs[0].metadata or {}).get("failed_buildings")
    assert recorded, "a failed run dropped the buildings it had already been refused"
    assert [r["bldg_id"] for r in recorded] == [str(doomed)]
    assert recorded[0]["puma_gisjoin"] == _PUMA

    # and the record is readable, so the next run skips that building
    assert failures.known_dead(root_uri, "resstock", _PUMA) == {doomed}


def test_a_permanently_missing_building_does_not_abort_the_puma(
    root_uri: str,
    offline_resstock: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    One absent file used to discard every completed request in the PUMA.

    Now the building is dropped and the PUMA is written short. The key does not
    say so -- it names the PUMA and nothing about what the fetch returned -- so
    what makes the shortfall visible is the pair that outlives any one run: the
    rows themselves, and the refusal recorded on the flow manifest, which is what
    a later run reads back to avoid asking again.
    """
    ids = sorted(_fixture("resstock_timeseries_bronze")["bldg_id"].unique().to_list())
    doomed = ids[0]
    monkeypatch.setattr(resstock_bronze, "fetch_building_frame", _doomed_seam(doomed))

    ingest_resstock(geographies=_geographies(), root_uri=root_uri)

    back = resstock_bronze.read_puma_timeseries_bronze(
        silver.resstock_timeseries_request(_geography()), root_uri
    )
    assert doomed not in back["bldg_id"].to_list()
    assert back["bldg_id"].n_unique() == len(ids) - 1
    # one write, and the shortfall is in the data rather than the key
    writes = _all_writes(resstock_bronze.TIMESERIES_DATASET_NAME, root_uri)
    assert len(writes) == 1
    assert "n_buildings" not in json.loads(writes[0].params_json)
    # and the failure is on the ledger, with its ids as fields
    flow = query_flow_manifests(root_uri=root_uri, status=FlowStatus.COMPLETED)[0]
    recorded = flow.metadata["failed_buildings"]
    assert [r["bldg_id"] for r in recorded] == [str(doomed)]
    assert recorded[0]["permanent"] == "True"
    assert recorded[0]["puma_gisjoin"] == _PUMA


def test_a_recorded_building_is_not_asked_for_again(
    root_uri: str,
    offline_resstock: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The other half of recording: reading it back.

    The climate pipeline recorded failures for months before anything consumed
    them, so every run re-asked the provider for units it had already refused.
    Here it matters doubly -- without the read-back the short write's key would
    never match a later run's plan, so the PUMA would re-fetch forever.
    """
    ids = sorted(_fixture("resstock_timeseries_bronze")["bldg_id"].unique().to_list())
    doomed = ids[0]
    monkeypatch.setattr(resstock_bronze, "fetch_building_frame", _doomed_seam(doomed))
    ingest_resstock(geographies=_geographies(), root_uri=root_uri)
    before = len(_all_writes(resstock_bronze.TIMESERIES_DATASET_NAME, root_uri))

    asked: list[int] = []
    original = resstock_bronze.fetch_building_frame

    def _counting(args, bldg_id, client):
        asked.append(bldg_id)
        return original(args, bldg_id, client)

    monkeypatch.setattr(resstock_bronze, "fetch_building_frame", _counting)
    ingest_resstock(geographies=_geographies(), root_uri=root_uri)

    assert doomed not in asked, "re-asked for a building already refused"
    # the short write is covered now, so nothing was fetched or written at all
    assert asked == []
    assert len(_all_writes(resstock_bronze.TIMESERIES_DATASET_NAME, root_uri)) == before


def test_a_transient_failure_does_not_write_a_short_puma(
    root_uri: str,
    offline_resstock: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A blip must not be recorded as a hole.

    Writing the PUMA short on a transient failure would make it indistinguishable
    from a permanent gap, and no later run would look for the missing building
    again. So the PUMA fails instead and is retried whole -- which now costs every
    fetch it had made, where a chunk once bounded that to 50.
    """
    ids = sorted(_fixture("resstock_timeseries_bronze")["bldg_id"].unique().to_list())
    # no retry delay: the point here is the outcome, not the backoff
    monkeypatch.setattr(
        flows,
        "fetch_resstock_building",
        flows.fetch_resstock_building.with_options(retries=0),
    )
    monkeypatch.setattr(
        resstock_bronze, "fetch_building_frame", _doomed_seam(ids[0], transient=True)
    )

    with pytest.raises(PipelineError, match="transient trouble"):
        ingest_resstock(geographies=_geographies(), root_uri=root_uri)

    # nothing was written for the PUMA, and nothing was recorded dead
    assert not _all_writes(resstock_bronze.TIMESERIES_DATASET_NAME, root_uri)
    failed_runs = query_flow_manifests(root_uri=root_uri, status=FlowStatus.FAILED)
    assert failed_runs and not (failed_runs[0].metadata or {}).get("failed_buildings")


def test_a_short_puma_is_refused_when_the_reader_asks_for_completeness(
    root_uri: str,
    offline_resstock: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The end of the chain #7 exists to close.

    #14 lets a PUMA complete over a building the release cannot serve, which is the
    right behaviour and also the moment the data starts understating demand. So the
    numbers have to meet: the write's key says what was written, the metadata
    bronze says what the PUMA has, and a reader that cares can refuse the
    difference.
    """
    ids = sorted(_fixture("resstock_timeseries_bronze")["bldg_id"].unique().to_list())
    monkeypatch.setattr(resstock_bronze, "fetch_building_frame", _doomed_seam(ids[0]))
    ingest_resstock(geographies=_geographies(), root_uri=root_uri)

    params = silver.resstock_timeseries_request(_geography())
    # the denominator comes from the metadata bronze the same run wrote
    expected = len(
        resstock_bronze.building_ids(
            resstock_bronze.read_metadata_bronze(
                silver.resstock_metadata_request(_geography()), root_uri
            ),
            _PUMA,
        )
    )
    assert expected == len(ids)

    # a reader that does not ask still gets the data, short and unannounced
    lenient = resstock_bronze.read_puma_timeseries_bronze(params, root_uri)
    assert lenient["bldg_id"].n_unique() == expected - 1

    # one that does ask is refused rather than handed an understated aggregate
    with pytest.raises(PipelineValueError, match="understates demand"):
        resstock_bronze.read_puma_timeseries_bronze(
            params,
            root_uri,
            expected_buildings=expected,
            require_complete=True,
        )


def test_each_building_fetch_takes_both_provider_limits(
    root_uri: str, offline_resstock: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Both budgets are acquired, and inside the task body.

    Two limits because they bound different things: one caps how many requests are
    in flight, the other how fast they start. A thread pool could only ever express
    the first, which is why replacing it with a mapped task is what made the second
    possible at all. Taken inside the body so a retry re-acquires rather than
    bypasses them.

    **Every** OEDI request answers to them, the metadata download included. That is
    what makes it safe to submit the metadata fetches together rather than walk
    them: the budget, not the loop, is what bounds how many fly at once.
    """
    taken: list[str] = []
    weights: list[int] = []

    class _Slot:
        def __init__(self, name: str, occupy: int) -> None:
            self.name = name
            self.occupy = occupy

        def __enter__(self):
            taken.append(f"concurrency:{self.name}")
            weights.append(self.occupy)
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    monkeypatch.setattr(
        flows, "concurrency", lambda name, occupy=1, **kw: _Slot(name, occupy)
    )
    monkeypatch.setattr(
        flows, "rate_limit", lambda name, **kw: taken.append(f"rate:{name}")
    )

    ingest_resstock(geographies=_geographies(), root_uri=root_uri)

    assert f"concurrency:{flows.OEDI_LIMIT}" in taken
    assert f"rate:{flows.OEDI_RATE_LIMIT}" in taken
    # one of each per request, and the in-flight slot is taken first
    assert taken.index(f"concurrency:{flows.OEDI_LIMIT}") < taken.index(
        f"rate:{flows.OEDI_RATE_LIMIT}"
    )
    # Every building, plus the one state metadata file this run's single state
    # needs. The metadata download used to bypass the budget entirely.
    buildings = _fixture("resstock_timeseries_bronze")["bldg_id"].n_unique()
    state_files = 1
    assert taken.count(f"rate:{flows.OEDI_RATE_LIMIT}") == buildings + state_files

    # And it is weighted by what it costs. A ~50 MB state file taking one slot --
    # the same as a ~6.3 MB building -- would let a twenty-PUMA run put ~1 GB in
    # flight on a link the whole design treats as the ceiling.
    assert flows.RESSTOCK_METADATA_SLOTS in weights
    assert max(weights) == flows.RESSTOCK_METADATA_SLOTS


def test_a_run_records_the_geographies_it_was_for(
    root_uri: str, offline_all: None
) -> None:
    """
    Which PUMAs a run was for, on the flow manifest.

    The dataset keys carry the PUMA wherever the write is PUMA-grained. The one
    exception is the ResStock state metadata, whose write holds every PUMA in the
    state -- a PUMA in *that* key would claim something untrue and would refetch the
    same 54 MB file once per PUMA. So the run-level answer lives here, where it
    identifies the run without touching what decides reuse.

    An entry per PUMA, in order: with a list as the input this is the only record
    of which PUMAs a run covered, so it has to stay iterable.
    """
    run_load_pipeline(geographies=_geographies(), root_uri=root_uri)

    for flow in query_flow_manifests(root_uri=root_uri, status=FlowStatus.COMPLETED):
        geos = (flow.metadata or {}).get("geographies")
        assert geos, f"{flow.flow_name} recorded no geographies"
        assert [geo["puma_gisjoin"] for geo in geos] == [_PUMA]
        assert geos[0]["state"] == _STATE
        assert geos[0]["utc_offset_minutes"] == "-300"  # DC, EST, no DST


def test_a_puma_grained_write_carries_the_puma_in_its_key(
    root_uri: str, offline_all: None
) -> None:
    """
    Every building-stock key names the PUMA its write holds.

    A key describes what its write *contains*, and all four of these now contain
    one PUMA -- the ResStock metadata included, which used to hold a whole state.
    That is what makes the manifest a PUMA index the way the climate pipeline's is
    a point index: one thing to ask for, one entry per thing.
    """
    run_load_pipeline(geographies=_geographies(), root_uri=root_uri)

    for dataset_name in (
        resstock_bronze.METADATA_DATASET_NAME,
        resstock_bronze.TIMESERIES_DATASET_NAME,
        comstock_bronze.PUMA_METADATA_DATASET_NAME,
        comstock_bronze.TIMESERIES_DATASET_NAME,
    ):
        writes = _all_writes(dataset_name, root_uri)
        assert writes, dataset_name
        for row in writes:
            params = json.loads(row.params_json)
            assert params["puma_gisjoin"] == _PUMA
            assert params["state"] == _STATE


# --------------------------------------------------------------------------- #
# A PUMA the release does not have
# --------------------------------------------------------------------------- #


def test_one_absent_puma_does_not_kill_the_run(
    root_uri: str, offline_one_absent_puma: None
) -> None:
    """
    Two PUMAs asked for, one of them a code no release has: the real one is ingested
    end to end and the run completes.

    The failure this replaces cost the whole run. A bad code would surface in the
    metadata phase, raise, and take with it every PUMA that *was* real -- for the
    building-stock sources thousands of requests and the better part of an hour, for
    a typo in one entry of a list.
    """
    run_load_pipeline(geographies=_geographies(_PUMA, _ABSENT_PUMA), root_uri=root_uri)

    # The real PUMA has all four building-stock datasets, and only it.
    for dataset_name in (
        resstock_bronze.METADATA_DATASET_NAME,
        resstock_bronze.TIMESERIES_DATASET_NAME,
        comstock_bronze.PUMA_METADATA_DATASET_NAME,
        comstock_bronze.TIMESERIES_DATASET_NAME,
    ):
        assert _pumas_written(dataset_name, root_uri) == {_PUMA}, dataset_name

    # The state-grained half still ran: its state was reached by a real PUMA.
    for source in dsgrid_bronze.PROFILES:
        assert query_manifest(
            dataset_name=dsgrid_bronze.PROFILES[source].timeseries_dataset_name,
            root_uri=root_uri,
        )
    for name in (silver.METADATA_DATASET_NAME, silver.TIMESERIES_DATASET_NAME):
        assert len(query_manifest(dataset_name=name, root_uri=root_uri)) == 1


def test_an_absent_puma_is_recorded_on_the_flow_manifest(
    root_uri: str, offline_one_absent_puma: None
) -> None:
    """
    A skipped PUMA is recorded, not merely logged.

    A log line is gone as soon as the run scrolls past; the manifest is what an
    operator reads afterwards to find out why a run they asked for two PUMAs of
    returned one.
    """
    run_load_pipeline(geographies=_geographies(_PUMA, _ABSENT_PUMA), root_uri=root_uri)

    parent = next(
        f
        for f in query_flow_manifests(root_uri=root_uri, status=FlowStatus.COMPLETED)
        if f.flow_name == "run_load_pipeline"
    )
    records = _rejected(parent)
    assert {entry["puma_gisjoin"] for entry in records} == {_ABSENT_PUMA}
    # Both releases refused it, each in its own way, and each says so for itself --
    # one source's gap is not evidence about the other's.
    assert {entry["source"] for entry in records} == {"resstock", "comstock"}
    assert all(entry["permanent"] == "True" for entry in records)

    # The run still records what it was *asked* for, so the skip is visible as a
    # difference rather than by the request quietly shrinking.
    assert {g["puma_gisjoin"] for g in parent.metadata["geographies"]} == {
        _PUMA,
        _ABSENT_PUMA,
    }


def test_a_run_of_only_absent_pumas_still_fails(
    root_uri: str, offline_one_absent_puma: None
) -> None:
    """
    Tolerance is for finishing the work that exists. With every code bad there is no
    run to save, and completing would report success for a run that wrote nothing.
    """
    with pytest.raises(PipelineValueError, match="every requested PUMA was rejected"):
        run_load_pipeline(geographies=_geographies(_ABSENT_PUMA), root_uri=root_uri)

    failed = query_flow_manifests(root_uri=root_uri, status=FlowStatus.FAILED)
    parent = next(f for f in failed if f.flow_name == "run_load_pipeline")
    assert {entry["puma_gisjoin"] for entry in _rejected(parent)} == {_ABSENT_PUMA}
    # Nothing was written for it, least of all an empty dataset under its key.
    assert not query_manifest(
        dataset_name=resstock_bronze.METADATA_DATASET_NAME, root_uri=root_uri
    )


def test_ingest_resstock_alone_tolerates_an_absent_puma(
    root_uri: str, offline_one_absent_puma: None
) -> None:
    """The single-source flows get the same treatment as the pipeline."""
    ingest_resstock(geographies=_geographies(_PUMA, _ABSENT_PUMA), root_uri=root_uri)

    assert _pumas_written(resstock_bronze.TIMESERIES_DATASET_NAME, root_uri) == {_PUMA}
    completed = next(
        f
        for f in query_flow_manifests(root_uri=root_uri, status=FlowStatus.COMPLETED)
        if f.flow_name == "ingest_resstock"
    )
    assert {entry["puma_gisjoin"] for entry in _rejected(completed)} == {_ABSENT_PUMA}


def test_a_rejection_is_per_source_not_pooled(
    root_uri: str,
    monkeypatch: pytest.MonkeyPatch,
    offline_dsgrid: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    A code one release has no file for is no evidence about the other, so only that
    release's half is dropped.

    ResStock and ComStock are separate releases with separately published geography.
    Pooling the rejections would let one release's gap silently discard data the
    other does publish -- an outcome worse than the failure this replaces, because
    it looks like success.
    """
    resstock_metadata = _fixture("resstock_metadata_bronze")  # holds _PUMA only
    comstock_timeseries = _fixture("comstock_timeseries_bronze")
    comstock_metadata = _relabelled(_collapsed_puma_metadata(), _SECOND_PUMA)

    monkeypatch.setattr(resstock_bronze, "download_object", lambda *a, **k: b"")
    monkeypatch.setattr(
        resstock_bronze, "parquet_to_metadata_table", lambda *a, **k: resstock_metadata
    )
    monkeypatch.setattr(comstock_bronze, "download_object", lambda url, **k: b"")
    monkeypatch.setattr(
        comstock_bronze,
        "parquet_to_puma_metadata_table",
        lambda *a, **k: comstock_metadata,
    )
    monkeypatch.setattr(
        comstock_bronze,
        "fetch_building_frame",
        lambda args, bldg_id, client: _building_frame(
            comstock_timeseries, args, bldg_id
        ),
    )

    # _SECOND_PUMA is absent from the residential state file but published by the
    # commercial release.
    assert _SECOND_PUMA not in resstock_metadata["puma_gisjoin"].to_list()

    with caplog.at_level(logging.WARNING):
        run_load_pipeline(geographies=_geographies(_SECOND_PUMA), root_uri=root_uri)

    # And the operator is told which of the two it is. Reporting a PUMA the run
    # ingested as "skipped" would send someone hunting a typo in a code that is
    # perfectly good -- the release simply differs from its neighbour.
    assert "ingested from that source alone" in caplog.text
    assert "were skipped" not in caplog.text

    # The commercial half ran for it...
    assert _pumas_written(comstock_bronze.PUMA_METADATA_DATASET_NAME, root_uri) == {
        _SECOND_PUMA
    }
    assert _pumas_written(comstock_bronze.TIMESERIES_DATASET_NAME, root_uri) == {
        _SECOND_PUMA
    }
    # ...and the residential half did not, because its release has no such code.
    assert not query_manifest(
        dataset_name=resstock_bronze.TIMESERIES_DATASET_NAME, root_uri=root_uri
    )

    parent = next(
        f
        for f in query_flow_manifests(root_uri=root_uri, status=FlowStatus.COMPLETED)
        if f.flow_name == "run_load_pipeline"
    )
    records = _rejected(parent)
    assert {entry["source"] for entry in records} == {"resstock"}


# --------------------------------------------------------------------------- #
# A run stopped from outside
# --------------------------------------------------------------------------- #


class _Interrupted(BaseException):
    """
    Stands in for Prefect's ``TerminationSignal``.

    Not the real thing, deliberately. Raising a ``TerminationSignal`` in-process
    makes Prefect's engine re-raise the signal at the interpreter, which kills the
    test session (pytest exits 143). What matters here is the *class* of exit --
    a ``BaseException``, which ``except Exception`` is not allowed to catch -- and
    that is exactly what this reproduces. ``TerminationSignal`` derives from
    ``ExternalSignal(BaseException)`` for the same reason.
    """


def _interrupt(*args: object, **kwargs: object) -> None:
    """Stop a run the way cancelling it does: partway, from outside."""
    raise _Interrupted


class _InterruptedTask:
    """A fetch task that is cancelled the moment the run submits it.

    ``run_load_pipeline`` submits its bronze tasks itself rather than going
    through ``_ingest_each_unit``, so this is the seam an interruption lands on
    there -- once the metadata phases have run and their records exist.
    """

    def submit(self, *args: object, **kwargs: object) -> None:
        raise _Interrupted


def test_a_cancelled_run_still_records_what_it_learned(
    root_uri: str, offline_one_absent_puma: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Cancelling a run must not lose the records it had already built.

    Prefect cancels by sending SIGTERM, which its engine turns into a
    ``TerminationSignal`` -- a ``BaseException``. The ``except Exception`` that
    records a failure never sees it, so before the ``finally`` the manifest was left
    saying "running" forever and the rejected PUMAs died in memory with the process.

    Interrupted *after* the metadata phases, which is where the records exist: the
    interruption stands in for an operator hitting cancel once the long
    per-building part starts.
    """
    monkeypatch.setattr(flows, "fetch_resstock_bronze", _InterruptedTask())

    with pytest.raises(_Interrupted):
        run_load_pipeline(
            geographies=_geographies(_PUMA, _ABSENT_PUMA), root_uri=root_uri
        )

    manifests = query_flow_manifests(root_uri=root_uri, flow_name="run_load_pipeline")
    assert len(manifests) == 1
    run = manifests[0]
    # Terminal, and honest about which ending it was: nothing failed, it was stopped.
    assert run.status == FlowStatus.CANCELLED
    assert run.end_time is not None
    # And the records survived, which is the whole point.
    assert {entry["puma_gisjoin"] for entry in _rejected(run)} == {_ABSENT_PUMA}


def test_an_interrupted_run_is_not_reported_as_failed(
    root_uri: str, offline_one_absent_puma: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A cancelled run is its own outcome, not a failure.

    Folding it into FAILED would put every operator-cancelled run in the bucket an
    engineer scans for real breakage -- and RUNNING, which is what it used to be
    left as, claims work is still happening in a process that has exited.
    """
    monkeypatch.setattr(flows, "_ingest_each_unit", _interrupt)
    with pytest.raises(_Interrupted):
        ingest_resstock(geographies=_geographies(), root_uri=root_uri)

    assert not query_flow_manifests(root_uri=root_uri, status=FlowStatus.FAILED)
    assert not query_flow_manifests(root_uri=root_uri, status=FlowStatus.RUNNING)
    assert (
        len(query_flow_manifests(root_uri=root_uri, status=FlowStatus.CANCELLED)) == 1
    )


def test_a_normal_ending_is_not_recorded_as_cancelled(
    root_uri: str, offline_all: None
) -> None:
    """The finally must not overwrite a run that ended on its own terms."""
    run_load_pipeline(geographies=_geographies(), root_uri=root_uri)

    assert not query_flow_manifests(root_uri=root_uri, status=FlowStatus.CANCELLED)
    completed = query_flow_manifests(root_uri=root_uri, status=FlowStatus.COMPLETED)
    assert {f.flow_name for f in completed} == {
        "run_load_pipeline",
        "industrial_load_silver",
    }


def test_a_failing_run_is_still_recorded_as_failed(
    root_uri: str, offline_one_absent_puma: None
) -> None:
    """...and neither must it relabel a genuine failure."""
    with pytest.raises(PipelineValueError, match="every requested PUMA was rejected"):
        run_load_pipeline(geographies=_geographies(_ABSENT_PUMA), root_uri=root_uri)

    assert not query_flow_manifests(root_uri=root_uri, status=FlowStatus.CANCELLED)
    failed = query_flow_manifests(root_uri=root_uri, status=FlowStatus.FAILED)
    assert [f.flow_name for f in failed] == ["run_load_pipeline"]


# --------------------------------------------------------------------------- #
# The opening phase overlaps too
# --------------------------------------------------------------------------- #


def test_the_metadata_fetches_overlap(
    root_uri: str, offline_two_pumas: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The metadata phase must fan out, not queue.

    It used to be a walk: one state file, then the next, then a request per PUMA,
    with nothing else able to start behind it. Independent downloads, so a
    twenty-PUMA run spent twenty round-trips before its first building was fetched.

    Seamed on the *commercial* parse only, because that is the phase where this run
    has two tasks to overlap: ComStock publishes per PUMA, so two PUMAs are two
    fetches. The residential file is per state and both fixture PUMAs are in DC, so
    that phase submits a single task -- instrumenting it would add a participant
    with nobody to meet, which is what made an earlier version of this test flaky
    (it had to guess a window wide enough for the pair and short enough not to
    stall on the loner). Both phases got the same change; one of them is testable
    here.

    A barrier, so the assertion is simultaneity rather than "two calls happened":
    a flag set on the way in stays set, and a walk would satisfy it long after the
    first fetch returned.
    """
    barrier = threading.Barrier(2, timeout=30)
    met: list[bool] = []
    real_comstock = comstock_bronze.parquet_to_puma_metadata_table

    def _comstock_seam(*args, **kwargs):
        try:
            barrier.wait()
            met.append(True)
        except threading.BrokenBarrierError:
            # Nobody else was inside the seam within the timeout, or the barrier
            # broke on an earlier lone arrival: these fetches are being walked.
            met.append(False)
        return real_comstock(*args, **kwargs)

    monkeypatch.setattr(
        comstock_bronze, "parquet_to_puma_metadata_table", _comstock_seam
    )

    run_load_pipeline(geographies=_geographies(_PUMA, _SECOND_PUMA), root_uri=root_uri)

    assert met and all(met), (
        f"the two commercial metadata fetches never met ({met}) -- the phase is "
        "walking them again"
    )


def test_a_state_file_is_still_downloaded_once_per_state(
    root_uri: str, offline_two_pumas: list[str]
) -> None:
    """
    Submitting the states together must not undo the hoist.

    The residential file is per state and both fixture PUMAs share one, so a run
    covering both still pays for exactly one ~54 MB download. Concurrency is meant
    to overlap independent work, not to turn one download into two.
    """
    run_load_pipeline(geographies=_geographies(_PUMA, _SECOND_PUMA), root_uri=root_uri)

    metadata_urls = [u for u in offline_two_pumas if "metadata_and_annual_results" in u]
    assert len(metadata_urls) == 1, metadata_urls
    # ...and both PUMAs still got their own write out of it.
    assert _pumas_written(resstock_bronze.METADATA_DATASET_NAME, root_uri) == {
        _PUMA,
        _SECOND_PUMA,
    }
