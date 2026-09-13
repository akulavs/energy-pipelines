"""
Tests for the ResStock bronze ingestion pipeline (metadata + PUMA timeseries)
"""

from __future__ import annotations

import datetime as dt
import json
import functools
import io
from pathlib import Path

import httpx
import polars as pl
import pydantic
import pytest
import respx
from polars.testing import assert_frame_equal

from common.exceptions import PipelineError, PipelineValueError
from common.storage import columnar
from common.storage.manifest import query_manifest
from external_data.load_pipeline import failures
from external_data.load_pipeline.resstock import bronze

FIXTURES = Path(__file__).parent / "fixtures"
METADATA_SAMPLE = FIXTURES / "resstock_metadata_sample.parquet"
METADATA_GOLDEN = FIXTURES / "resstock_metadata_bronze.parquet"
# Reused by the PUMA tests to synthesize per-building timeseries responses
TIMESERIES_SAMPLE = FIXTURES / "resstock_timeseries_sample.parquet"

# Real DC baseline slice projected to the curated columns
_METADATA_ROWS = 261

_FIXED_WRITE_TIME = dt.datetime(2026, 7, 14, tzinfo=dt.timezone.utc)


@pytest.fixture(scope="session")
def metadata_bytes() -> bytes:
    return METADATA_SAMPLE.read_bytes()


@pytest.fixture(scope="session")
def metadata_golden() -> pl.DataFrame:
    return pl.read_parquet(METADATA_GOLDEN)


# --------------------------------------------------------------------------- #
# Request / provenance
# --------------------------------------------------------------------------- #


def test_metadata_url_matches_params() -> None:
    args = bronze.ResstockMetadataRequestArgs()
    assert args.source_dataset == bronze.DATASET
    # ResStock metadata is partitioned by state only
    assert args.object_url() == (
        f"{bronze.BASE_URL}/{bronze.OEDI_PREFIX}/2025/resstock_amy2018_release_1/"
        "metadata_and_annual_results/by_state/full/parquet/"
        "state=DC/DC_upgrade0.parquet"
    )


@pytest.mark.parametrize("state", ["district", "d", "dc", "USA"])
def test_bad_state_rejected(state: str) -> None:
    with pytest.raises(pydantic.ValidationError):
        bronze.ResstockMetadataRequestArgs(state=state)


def test_negative_upgrade_rejected() -> None:
    with pytest.raises(pydantic.ValidationError):
        bronze.ResstockMetadataRequestArgs(upgrade=-1)


# --------------------------------------------------------------------------- #
# Fetch
# --------------------------------------------------------------------------- #


@respx.mock
def test_download_object_hits_expected_endpoint(metadata_bytes: bytes) -> None:
    url = bronze.ResstockMetadataRequestArgs().object_url()
    route = respx.get(url).mock(
        return_value=httpx.Response(200, content=metadata_bytes)
    )
    body = bronze.download_object(url)
    assert body == metadata_bytes
    assert route.called
    assert str(route.calls.last.request.url) == url


@respx.mock
def test_download_object_4xx_raises() -> None:
    url = bronze.ResstockMetadataRequestArgs(state="ZZ").object_url()
    respx.get(url).mock(return_value=httpx.Response(403, text="AccessDenied"))
    with pytest.raises(
        failures.PermanentFetchError, match="OEDI cannot serve this request"
    ):
        bronze.download_object(url, max_retries=1)


@respx.mock
def test_download_object_retries_5xx_then_succeeds(metadata_bytes: bytes) -> None:
    url = bronze.ResstockMetadataRequestArgs().object_url()
    route = respx.get(url).mock(
        side_effect=[
            httpx.Response(503, text="SlowDown"),
            httpx.Response(200, content=metadata_bytes),
        ]
    )
    body = bronze.download_object(url, max_retries=3, backoff_seconds=0)
    assert body == metadata_bytes
    assert route.call_count == 2


@respx.mock
def test_download_object_persistent_5xx_raises() -> None:
    url = bronze.ResstockMetadataRequestArgs().object_url()
    route = respx.get(url).mock(return_value=httpx.Response(500, text="oops"))
    with pytest.raises(PipelineError, match="after 3 attempts"):
        bronze.download_object(url, max_retries=3, backoff_seconds=0)
    assert route.call_count == 3


@respx.mock
def test_download_object_retries_request_error(metadata_bytes: bytes) -> None:
    url = bronze.ResstockMetadataRequestArgs().object_url()
    route = respx.get(url).mock(
        side_effect=[
            httpx.ConnectError("connection reset"),
            httpx.Response(200, content=metadata_bytes),
        ]
    )
    body = bronze.download_object(url, max_retries=3, backoff_seconds=0)
    assert body == metadata_bytes
    assert route.call_count == 2


@respx.mock
def test_download_object_persistent_request_error_raises() -> None:
    url = bronze.ResstockMetadataRequestArgs().object_url()
    respx.get(url).mock(side_effect=httpx.ConnectError("connection reset"))
    with pytest.raises(PipelineError, match="after 2 attempts"):
        bronze.download_object(url, max_retries=2, backoff_seconds=0)


# --------------------------------------------------------------------------- #
# Parse -- metadata
# --------------------------------------------------------------------------- #


def test_metadata_parse_produces_curated_columns(metadata_bytes: bytes) -> None:
    table = bronze.parquet_to_metadata_table(metadata_bytes)
    assert table.columns == bronze.METADATA_COLUMNS
    assert table.height == _METADATA_ROWS
    assert not any(table[c].null_count() == table.height for c in table.columns)


def test_metadata_matches_golden(
    metadata_bytes: bytes, metadata_golden: pl.DataFrame
) -> None:
    table = bronze.parquet_to_metadata_table(metadata_bytes)
    assert_frame_equal(
        table.sort(bronze.METADATA_KEY_COLUMNS),
        metadata_golden.sort(bronze.METADATA_KEY_COLUMNS),
    )


def test_metadata_is_one_row_per_building(metadata_bytes: bytes) -> None:
    # ResStock has no census-tract split (unlike ComStock): bldg_id is unique.
    table = bronze.parquet_to_metadata_table(metadata_bytes)
    assert table["bldg_id"].n_unique() == table.height
    assert table.select(bronze.METADATA_KEY_COLUMNS).is_duplicated().sum() == 0
    # weight is a single uniform value per state
    assert table["weight"].n_unique() == 1


def test_metadata_preserves_spaced_titlecase_and_none_literal(
    metadata_bytes: bytes,
) -> None:
    table = bronze.parquet_to_metadata_table(metadata_bytes)
    fuels = table["heating_fuel"].unique().to_list()
    # Spaced Title Case + literal "None"
    assert "Natural Gas" in fuels
    assert "None" in fuels
    assert "None" in table["hvac_cooling_type"].unique().to_list()


def test_metadata_fails_loud_on_missing_column(metadata_bytes: bytes) -> None:
    stripped = pl.read_parquet(metadata_bytes).drop("in.sqft..ft2")
    buf = io.BytesIO()
    stripped.write_parquet(buf)
    with pytest.raises(PipelineValueError, match="missing expected column"):
        bronze.parquet_to_metadata_table(buf.getvalue())


def test_metadata_dtypes(metadata_bytes: bytes) -> None:
    table = bronze.parquet_to_metadata_table(metadata_bytes)
    bronze.ResstockMetadataBronzeSchema.validate(table)
    assert table.schema["bldg_id"] == pl.Int64
    assert table.schema["weight"] == pl.Float64
    assert table.schema["annual_electricity_cooling_kwh"] == pl.Float64


# --------------------------------------------------------------------------- #
# Write + manifest + partitioning
# --------------------------------------------------------------------------- #


def test_write_metadata_bronze_round_trips_unpartitioned(
    metadata_bytes: bytes, tmp_path: Path
) -> None:
    params = bronze.ResstockMetadataRequestArgs()
    table = bronze.parquet_to_metadata_table(metadata_bytes)
    row = bronze.write_metadata_bronze(
        table, params, tmp_path, write_time=_FIXED_WRITE_TIME
    )

    assert row.dataset_name == bronze.METADATA_DATASET_NAME
    assert row.writer == bronze.DEFAULT_WRITER
    assert row.write_time == _FIXED_WRITE_TIME
    assert (
        bronze.ResstockMetadataRequestArgs.model_validate_json(row.params_json)
        == params
    )

    # No partition directories: the write is one PUMA and the manifest is the
    # index, as in the climate pipeline's per-point writes. A county partition
    # would scatter one PUMA across the counties it overlaps.
    assert not list(tmp_path.rglob("county_gisjoin=*"))
    assert len(list(tmp_path.rglob("_manifests/**/*.json"))) == 1

    back = columnar.read_dataset(
        bronze.ResstockMetadataBronzeSchema,
        bronze.METADATA_DATASET_NAME,
        params,
        str(tmp_path),
    )
    assert_frame_equal(
        back.sort(bronze.METADATA_KEY_COLUMNS).select(table.columns),
        table.sort(bronze.METADATA_KEY_COLUMNS),
        check_row_order=True,
    )


@respx.mock
def test_ingest_metadata_bronze_offline_end_to_end(
    metadata_bytes: bytes, tmp_path: Path
) -> None:
    params = bronze.ResstockMetadataRequestArgs()
    respx.get(params.object_url()).mock(
        return_value=httpx.Response(200, content=metadata_bytes)
    )
    row = bronze.ingest_metadata_bronze(params, root_uri=tmp_path)
    assert row.dataset_name == bronze.METADATA_DATASET_NAME

    back = columnar.read_dataset(
        bronze.ResstockMetadataBronzeSchema,
        bronze.METADATA_DATASET_NAME,
        params,
        str(tmp_path),
    )
    # The state file holds every PUMA in DC; the write holds the one its key names.
    assert back["puma_gisjoin"].unique().to_list() == [params.puma_gisjoin]
    assert 0 < back.height < _METADATA_ROWS


@respx.mock
def test_ingest_metadata_for_several_pumas_downloads_the_state_file_once(
    metadata_bytes: bytes, tmp_path: Path
) -> None:
    """
    A per-PUMA key must not mean a download per PUMA.

    OEDI publishes this file per state, so a run covering several of a state's
    PUMAs reads it once and cuts a write from it for each -- 54 MB paid once for a
    large state, not once per key.
    """
    table = bronze.parquet_to_metadata_table(metadata_bytes)
    pumas = table["puma_gisjoin"].unique().sort().to_list()[:2]
    params = [bronze.ResstockMetadataRequestArgs(puma_gisjoin=puma) for puma in pumas]
    route = respx.get(params[0].object_url()).mock(
        return_value=httpx.Response(200, content=metadata_bytes)
    )

    rows, rejected = bronze.ingest_metadata_bronze_for_pumas(params, root_uri=tmp_path)

    assert route.call_count == 1
    assert not rejected
    assert len(rows) == len(pumas)
    for request, puma in zip(params, pumas, strict=True):
        back = columnar.read_dataset(
            bronze.ResstockMetadataBronzeSchema,
            bronze.METADATA_DATASET_NAME,
            request,
            str(tmp_path),
        )
        assert back["puma_gisjoin"].unique().to_list() == [puma]


def test_a_metadata_slice_for_an_absent_puma_fails_loud(
    metadata_bytes: bytes,
) -> None:
    """
    An empty write under a PUMA's key is worse than no write: a later run finds it
    covered and never looks again.
    """
    table = bronze.parquet_to_metadata_table(metadata_bytes)
    with pytest.raises(PipelineValueError, match="no buildings found for PUMA"):
        bronze.metadata_for_puma(table, "G06000101")


def test_an_absent_puma_is_permanent_not_transient(metadata_bytes: bytes) -> None:
    """
    The state file has been read and has no such code, so a second look cannot
    change the answer. Permanence is what lets a caller drop the one PUMA instead of
    the run, and what stops the retry policy spending attempts on it.
    """
    table = bronze.parquet_to_metadata_table(metadata_bytes)
    with pytest.raises(failures.PermanentFetchError) as caught:
        bronze.metadata_for_puma(table, "G06000101")
    assert failures.is_permanent(caught.value)
    # Still a PipelineValueError, so callers that only wanted the loud failure are
    # unaffected by the narrowing.
    assert isinstance(caught.value, PipelineValueError)


@respx.mock
def test_one_absent_puma_does_not_cost_the_others_their_write(
    metadata_bytes: bytes, tmp_path: Path
) -> None:
    """
    The point of the whole change: an engineer's typo in a list of PUMAs takes its
    own PUMA out of the run and leaves the rest alone.

    The download is already paid for and the good PUMAs are sitting in the parsed
    frame, so raising here would throw away work that has already succeeded.
    """
    table = bronze.parquet_to_metadata_table(metadata_bytes)
    real = table["puma_gisjoin"].unique().sort().to_list()[0]
    absent = "G11009999"  # well-formed, same state, no such PUMA
    params = [
        bronze.ResstockMetadataRequestArgs(puma_gisjoin=real),
        bronze.ResstockMetadataRequestArgs(puma_gisjoin=absent),
    ]
    respx.get(params[0].object_url()).mock(
        return_value=httpx.Response(200, content=metadata_bytes)
    )

    rows, rejected = bronze.ingest_metadata_bronze_for_pumas(params, root_uri=tmp_path)

    # The real PUMA is written, and readable under its own key.
    assert [json.loads(row.params_json)["puma_gisjoin"] for row in rows] == [real]
    back = columnar.read_dataset(
        bronze.ResstockMetadataBronzeSchema,
        bronze.METADATA_DATASET_NAME,
        params[0],
        str(tmp_path),
    )
    assert back["puma_gisjoin"].unique().to_list() == [real]

    # The bad one comes back as a record, not an exception.
    assert len(rejected) == 1
    assert rejected[0]["puma_gisjoin"] == absent
    assert rejected[0]["source"] == "resstock"
    assert rejected[0]["permanent"] == "True"
    assert "no buildings found" in rejected[0]["error"]

    # And nothing was written under the bad key -- an empty dataset there is the
    # thing a later run would find "covered" and never look behind.
    assert not query_manifest(
        dataset_name=bronze.METADATA_DATASET_NAME,
        root_uri=str(tmp_path),
        params_where={"puma_gisjoin": absent},
    )


@respx.mock
def test_the_single_puma_door_still_raises(
    metadata_bytes: bytes, tmp_path: Path
) -> None:
    """
    Tolerance is only meaningful across a list. Asked about exactly one PUMA there is
    no run to save, so ``ingest_metadata_bronze`` keeps raising -- a caller that gets
    a row back can rely on it being that PUMA's.
    """
    args = bronze.ResstockMetadataRequestArgs(puma_gisjoin="G11009999")
    respx.get(args.object_url()).mock(
        return_value=httpx.Response(200, content=metadata_bytes)
    )
    with pytest.raises(failures.PermanentFetchError, match="no buildings found"):
        bronze.ingest_metadata_bronze(args, root_uri=tmp_path)


def test_metadata_requests_spanning_two_states_are_refused(tmp_path: Path) -> None:
    """One call reads one state's file; a mixed list would need a second fetch."""
    with pytest.raises(PipelineValueError, match="span"):
        bronze.ingest_metadata_bronze_for_pumas(
            [
                bronze.ResstockMetadataRequestArgs(
                    state="DC", puma_gisjoin="G11000101"
                ),
                bronze.ResstockMetadataRequestArgs(
                    state="DE", puma_gisjoin="G10000101"
                ),
            ],
            root_uri=tmp_path,
        )


# --------------------------------------------------------------------------- #
# Live integration (real OEDI data lake -- no credentials required)
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_ingest_metadata_bronze_live(tmp_path: Path) -> None:
    """
    Full pipeline against the real public OEDI data lake
    """
    params = bronze.ResstockMetadataRequestArgs()
    row = bronze.ingest_metadata_bronze(root_uri=tmp_path)
    assert row.dataset_name == bronze.METADATA_DATASET_NAME

    back = columnar.read_dataset(
        bronze.ResstockMetadataBronzeSchema,
        bronze.METADATA_DATASET_NAME,
        params,
        str(tmp_path),
    ).select(bronze.ResstockMetadataBronzeSchema.columns)

    bronze.ResstockMetadataBronzeSchema.validate(back)
    assert list(back.columns) == bronze.METADATA_COLUMNS
    # one row per building; cooling <= total everywhere
    assert back["bldg_id"].n_unique() == back.height
    assert (
        back["annual_electricity_cooling_kwh"] <= back["annual_electricity_total_kwh"]
    ).all()
    assert (back["annual_electricity_total_kwh"] >= 0).all()


# =========================================================================== #
# PUMA-batched timeseries (resstock_timeseries_bronze)
# =========================================================================== #

# A PUMA present in the metadata sample slice
_SAMPLE_PUMA = "G11000105"


@functools.lru_cache(maxsize=1)
def _one_building_raw_ts() -> pl.DataFrame:
    """A single building's raw (pre-curation) timeseries rows from the sample.

    Cached: the offline end-to-end test now synthesizes a response for every
    building in the PUMA, so this would otherwise re-read the fixture ~75 times.
    """
    raw = pl.read_parquet(TIMESERIES_SAMPLE)
    first = raw["bldg_id"].min()
    return raw.filter(pl.col("bldg_id") == first)


def _ts_bytes_for(bldg_id: int) -> bytes:
    """Synthesize a per-building raw timeseries parquet with the given bldg_id."""
    frame = _one_building_raw_ts().with_columns(
        pl.lit(bldg_id, dtype=pl.Int64).alias("bldg_id")
    )
    buf = io.BytesIO()
    frame.write_parquet(buf)
    return buf.getvalue()


def _ts_response(request: httpx.Request) -> httpx.Response:
    """Serve any per-building timeseries URL, deriving the id from the path.

    One route for the whole PUMA: the ingest fetches *every* building, so mocking
    them individually would mean one route per building and would quietly break
    whenever the fixture gains one.
    """
    bldg_id = int(request.url.path.rsplit("/", 1)[-1].split("-")[0])
    return httpx.Response(200, content=_ts_bytes_for(bldg_id))


_TS_URL_PATTERN = r".*/timeseries_individual_buildings/.*\.parquet$"


def _sample_puma_bldg_ids() -> list[int]:
    md = pl.read_parquet(METADATA_SAMPLE, columns=["bldg_id", "in.puma"])
    return [
        int(b)
        for b in md.filter(pl.col("in.puma") == _SAMPLE_PUMA)["bldg_id"]
        .unique()
        .sort()
        .to_list()
    ]


def test_puma_state_metadata_url_matches_params() -> None:
    args = bronze.ResstockPumaTimeseriesRequestArgs()
    assert args.state_metadata_url() == (
        f"{bronze.BASE_URL}/{bronze.OEDI_PREFIX}/2025/resstock_amy2018_release_1/"
        "metadata_and_annual_results/by_state/full/parquet/"
        "state=DC/DC_upgrade0.parquet"
    )


def test_puma_building_timeseries_url_matches_params() -> None:
    args = bronze.ResstockPumaTimeseriesRequestArgs()
    assert args.building_timeseries_url(100524) == (
        f"{bronze.BASE_URL}/{bronze.OEDI_PREFIX}/2025/resstock_amy2018_release_1/"
        "timeseries_individual_buildings/by_state/upgrade=0/"
        "state=DC/100524-0.parquet"
    )


@pytest.mark.parametrize("puma", ["G1100010", "11000105", "G110001050", "puma"])
def test_bad_puma_rejected(puma: str) -> None:
    with pytest.raises(pydantic.ValidationError):
        bronze.ResstockPumaTimeseriesRequestArgs(puma_gisjoin=puma)


@respx.mock
def test_resolve_puma_building_ids_filters_to_puma(metadata_bytes: bytes) -> None:
    args = bronze.ResstockPumaTimeseriesRequestArgs(puma_gisjoin=_SAMPLE_PUMA)
    respx.get(args.state_metadata_url()).mock(
        return_value=httpx.Response(200, content=metadata_bytes)
    )
    ids = bronze.resolve_puma_building_ids(args)
    assert ids == _sample_puma_bldg_ids()
    assert ids == sorted(set(ids))  # sorted + unique


@respx.mock
def test_resolve_puma_building_ids_fails_loud_on_empty(metadata_bytes: bytes) -> None:
    args = bronze.ResstockPumaTimeseriesRequestArgs(puma_gisjoin="G99999999")
    respx.get(args.state_metadata_url()).mock(
        return_value=httpx.Response(200, content=metadata_bytes)
    )
    with pytest.raises(PipelineValueError, match="no buildings found for PUMA"):
        bronze.resolve_puma_building_ids(args)


@respx.mock
def test_ingest_puma_timeseries_offline_end_to_end(
    metadata_bytes: bytes, tmp_path: Path
) -> None:
    args = bronze.ResstockPumaTimeseriesRequestArgs(puma_gisjoin=_SAMPLE_PUMA)
    respx.get(args.state_metadata_url()).mock(
        return_value=httpx.Response(200, content=metadata_bytes)
    )
    # A location means every building in it, so serve them all from one route.
    expected = _sample_puma_bldg_ids()
    respx.get(url__regex=_TS_URL_PATTERN).mock(side_effect=_ts_response)

    rows = bronze.ingest_puma_timeseries_bronze(
        args,
        root_uri=tmp_path,
        write_time=_FIXED_WRITE_TIME,
    )
    # One write for the whole PUMA, not one per 50-building chunk.
    assert len(rows) == 1
    assert rows[0].dataset_name == bronze.TIMESERIES_DATASET_NAME
    assert rows[0].writer == bronze.DEFAULT_WRITER
    # The key is the PUMA and nothing about what the fetch returned, so it is the
    # key a reader can build: no count to guess at.
    key = json.loads(rows[0].params_json)
    assert key["puma_gisjoin"] == _SAMPLE_PUMA
    assert "n_buildings" not in key, key
    assert bronze.ResstockPumaTimeseriesRequestArgs.model_validate_json(
        rows[0].params_json
    )

    # No partition directories: the write is one PUMA and its key says so.
    assert not list(tmp_path.rglob("puma_gisjoin=*"))
    assert len(list(tmp_path.rglob(f"{bronze.TIMESERIES_DATASET_NAME}/*.parquet"))) == 1

    # Read back through the same helper: one manifest scan, matched on the base
    # fields, because the key carries a building count the caller does not know.
    back = bronze.read_puma_timeseries_bronze(args, tmp_path)
    assert back.columns == bronze.TIMESERIES_COLUMNS
    # every building in the PUMA, assembled into one dataset with bldg_id as a
    # join-key column -- the ingest has no way to write a subset
    assert sorted(back["bldg_id"].unique().to_list()) == expected
    assert len(expected) > 2, "fixture should hold a real PUMA's worth of buildings"
    assert back["state"].unique().to_list() == ["DC"]
    assert back["puma_gisjoin"].unique().to_list() == [_SAMPLE_PUMA]


@pytest.mark.integration
def test_ingest_puma_timeseries_bronze_live(tmp_path: Path) -> None:
    """
    Fetch + shape + write against the real OEDI data lake, three buildings' worth
    """
    args = bronze.ResstockPumaTimeseriesRequestArgs()  # DC, PUMA G11000101
    # ``ingest_puma_timeseries_bronze`` fetches the whole PUMA (~200 requests),
    # which is more than a test should spend, so drive the same
    # resolve -> fetch -> write path with an explicit slice. Truncating *here* is
    # the point of the pipeline having no cap: only a test can produce a partial
    # write, and this one writes into tmp_path.
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        bldg_ids = bronze.resolve_puma_building_ids(args, client=client)[:3]
        table = bronze.fetch_puma_timeseries_table(args, client, bldg_ids)
    row = bronze.write_puma_timeseries_bronze(table, args, tmp_path)
    assert row.dataset_name == bronze.TIMESERIES_DATASET_NAME

    back = bronze.read_puma_timeseries_bronze(args, tmp_path).select(
        bronze.ResstockTimeseriesBronzeSchema.columns
    )

    bronze.ResstockTimeseriesBronzeSchema.validate(back)
    # three distinct buildings, each a full 2018 year at 15-min resolution
    assert back["bldg_id"].n_unique() == 3
    assert back.height == 3 * 365 * 24 * 4
    assert back["puma_gisjoin"].unique().to_list() == [args.puma_gisjoin]
    assert (back["electricity_cooling_kwh"] <= back["electricity_total_kwh"]).all()
