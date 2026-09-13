"""
Tests for the ComStock bronze ingestion pipeline (metadata + PUMA timeseries)
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

from common.exceptions import PipelineError, PipelineValueError
from external_data.load_pipeline import failures
from external_data.load_pipeline.comstock import bronze

FIXTURES = Path(__file__).parent / "fixtures"
METADATA_SAMPLE = FIXTURES / "comstock_metadata_sample.parquet"
METADATA_GOLDEN = FIXTURES / "comstock_metadata_bronze.parquet"
# Reused by the PUMA tests to synthesize per-building timeseries responses
TIMESERIES_SAMPLE = FIXTURES / "comstock_timeseries_sample.parquet"

# Real DC baseline slice (one row per building x census tract)
_METADATA_ROWS = 1661

# A PUMA present in the metadata sample slice
_SAMPLE_PUMA = "G11000101"

_FIXED_WRITE_TIME = dt.datetime(2026, 7, 14, tzinfo=dt.timezone.utc)


@pytest.fixture(scope="session")
def metadata_bytes() -> bytes:
    return METADATA_SAMPLE.read_bytes()


@functools.lru_cache(maxsize=1)
def _one_building_raw_ts() -> pl.DataFrame:
    """A single building's raw (pre-curation) timeseries rows from the sample.

    Cached: the offline end-to-end test now synthesizes a response for every
    building in the PUMA, so this would otherwise re-read the fixture ~65 times.
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


def _puma_metadata_bytes(puma: str = _SAMPLE_PUMA) -> bytes:
    """
    Synthesize the per-PUMA metadata file from the real per-county slice.

    Applies exactly the collapse that OEDI file embodies -- one row per building,
    with ``weight`` summed over the census tracts the model represents -- so the
    fixture encodes the invariant the ingest depends on rather than freezing a
    second snapshot that could drift from the first.
    """
    raw = pl.read_parquet(METADATA_SAMPLE).filter(
        pl.col("in.nhgis_puma_gisjoin") == puma
    )
    others = [
        c for c in bronze.PUMA_METADATA_RAW_TO_COLUMN if c not in ("bldg_id", "weight")
    ]
    collapsed = raw.group_by("bldg_id").agg(
        pl.col("weight").sum(),
        *[pl.col(c).first() for c in others],
    )
    buf = io.BytesIO()
    collapsed.write_parquet(buf)
    return buf.getvalue()


def _sample_puma_bldg_ids() -> list[int]:
    md = pl.read_parquet(METADATA_SAMPLE, columns=["bldg_id", "in.nhgis_puma_gisjoin"])
    return [
        int(b)
        for b in md.filter(pl.col("in.nhgis_puma_gisjoin") == _SAMPLE_PUMA)["bldg_id"]
        .unique()
        .sort()
        .to_list()
    ]


# --------------------------------------------------------------------------- #
# Request / provenance
# --------------------------------------------------------------------------- #


def test_puma_building_timeseries_url_matches_params() -> None:
    args = bronze.ComstockPumaTimeseriesRequestArgs()
    assert args.building_timeseries_url(10045) == (
        f"{bronze.BASE_URL}/{bronze.OEDI_PREFIX}/2025/comstock_amy2018_release_3/"
        "timeseries_individual_buildings/by_state/upgrade=0/"
        "state=DC/10045-0.parquet"
    )


def test_puma_metadata_url_matches_params() -> None:
    args = bronze.ComstockPumaMetadataRequestArgs()
    assert args.object_url() == (
        f"{bronze.BASE_URL}/{bronze.OEDI_PREFIX}/2025/comstock_amy2018_release_3/"
        "metadata_and_annual_results_aggregates/by_state_and_puma/full/parquet/"
        "state=DC/puma=G11000101/DC_G11000101_upgrade0_agg.parquet"
    )


@pytest.mark.parametrize("state", ["district", "d", "dc", "USA"])
def test_bad_state_rejected(state: str) -> None:
    with pytest.raises(pydantic.ValidationError):
        bronze.ComstockPumaMetadataRequestArgs(state=state)


def test_bad_county_and_puma_and_upgrade_rejected() -> None:
    with pytest.raises(pydantic.ValidationError):
        bronze.ComstockPumaMetadataRequestArgs(puma_gisjoin="G123")
    with pytest.raises(pydantic.ValidationError):
        bronze.ComstockPumaTimeseriesRequestArgs(puma_gisjoin="G1100010")  # 7 digits
    with pytest.raises(pydantic.ValidationError):
        bronze.ComstockPumaMetadataRequestArgs(upgrade=-1)


# --------------------------------------------------------------------------- #
# Fetch
# --------------------------------------------------------------------------- #


@respx.mock
def test_download_object_hits_expected_endpoint(metadata_bytes: bytes) -> None:
    url = bronze.ComstockPumaMetadataRequestArgs().object_url()
    route = respx.get(url).mock(
        return_value=httpx.Response(200, content=metadata_bytes)
    )
    body = bronze.download_object(url)
    assert body == metadata_bytes
    assert route.called


@respx.mock
def test_download_object_4xx_raises() -> None:
    url = bronze.ComstockPumaMetadataRequestArgs(state="ZZ").object_url()
    respx.get(url).mock(return_value=httpx.Response(403, text="AccessDenied"))
    with pytest.raises(
        failures.PermanentFetchError, match="OEDI cannot serve this request"
    ):
        bronze.download_object(url, max_retries=1)


@respx.mock
def test_download_object_retries_5xx_then_succeeds(metadata_bytes: bytes) -> None:
    url = bronze.ComstockPumaMetadataRequestArgs().object_url()
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
    url = bronze.ComstockPumaMetadataRequestArgs().object_url()
    route = respx.get(url).mock(return_value=httpx.Response(500, text="oops"))
    with pytest.raises(PipelineError, match="after 3 attempts"):
        bronze.download_object(url, max_retries=3, backoff_seconds=0)
    assert route.call_count == 3


@respx.mock
def test_download_object_retries_request_error(metadata_bytes: bytes) -> None:
    url = bronze.ComstockPumaMetadataRequestArgs().object_url()
    route = respx.get(url).mock(
        side_effect=[
            httpx.ConnectError("connection reset"),
            httpx.Response(200, content=metadata_bytes),
        ]
    )
    body = bronze.download_object(url, max_retries=3, backoff_seconds=0)
    assert body == metadata_bytes
    assert route.call_count == 2


# --------------------------------------------------------------------------- #
# Parse -- metadata
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Resolve PUMA -> building ids
# --------------------------------------------------------------------------- #


@respx.mock
def test_resolve_puma_building_ids_reads_one_puma_file() -> None:
    """
    One request, against a file keyed by the PUMA being asked about.

    This replaced a whole-state scan: the per-county files carry no PUMA index, so
    every county file in the state was downloaded and filtered, once per PUMA.
    """
    args = bronze.ComstockPumaTimeseriesRequestArgs(puma_gisjoin=_SAMPLE_PUMA)
    route = respx.get(bronze.puma_metadata_request(args).object_url()).mock(
        return_value=httpx.Response(200, content=_puma_metadata_bytes())
    )
    ids = bronze.resolve_puma_building_ids(args)
    assert route.call_count == 1
    assert ids == _sample_puma_bldg_ids()
    assert ids == sorted(set(ids))


@respx.mock
def test_resolve_puma_building_ids_fails_loud_on_an_empty_file() -> None:
    """
    A file that exists but holds no buildings is still a bad pairing.

    A PUMA that does not exist now 404s instead, which ``download_object`` reports
    as a caller error -- better feedback than the old silent empty scan.
    """
    args = bronze.ComstockPumaTimeseriesRequestArgs(puma_gisjoin="G99999999")
    empty = pl.read_parquet(METADATA_SAMPLE).head(0)
    buf = io.BytesIO()
    empty.write_parquet(buf)
    respx.get(bronze.puma_metadata_request(args).object_url()).mock(
        return_value=httpx.Response(200, content=buf.getvalue())
    )
    with pytest.raises(PipelineValueError, match="no buildings found for PUMA"):
        bronze.resolve_puma_building_ids(args)


# --------------------------------------------------------------------------- #
# Fetch + assemble + write (offline end-to-end)
# --------------------------------------------------------------------------- #


@respx.mock
def test_ingest_puma_timeseries_offline_end_to_end(
    metadata_bytes: bytes, tmp_path: Path
) -> None:
    args = bronze.ComstockPumaTimeseriesRequestArgs(puma_gisjoin=_SAMPLE_PUMA)
    respx.get(bronze.puma_metadata_request(args).object_url()).mock(
        return_value=httpx.Response(200, content=_puma_metadata_bytes())
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
    assert bronze.ComstockPumaTimeseriesRequestArgs.model_validate_json(
        rows[0].params_json
    )

    # No partition directories: the write is one PUMA and its key says so.
    assert not list(tmp_path.rglob("puma_gisjoin=*"))
    assert len(list(tmp_path.rglob(f"{bronze.TIMESERIES_DATASET_NAME}/*.parquet"))) == 1

    # Read back through the same helper: one manifest scan, matched on the base
    # fields, because the key carries a building count the caller does not know.
    back = bronze.read_puma_timeseries_bronze(args, tmp_path)
    assert back.columns == bronze.TIMESERIES_COLUMNS
    # every building in the PUMA -- the ingest has no way to write a subset
    assert sorted(back["bldg_id"].unique().to_list()) == expected
    assert len(expected) > 2, "fixture should hold a real PUMA's worth of buildings"
    assert back["state"].unique().to_list() == ["DC"]
    assert back["puma_gisjoin"].unique().to_list() == [_SAMPLE_PUMA]


def test_fetch_puma_timeseries_fails_loud_on_empty_bldg_ids() -> None:
    args = bronze.ComstockPumaTimeseriesRequestArgs(puma_gisjoin=_SAMPLE_PUMA)
    with httpx.Client() as client:
        with pytest.raises(PipelineValueError, match="empty bldg_ids"):
            bronze.fetch_puma_timeseries_table(args, client, bldg_ids=[])


@respx.mock
def test_fetch_puma_timeseries_wraps_building_failure() -> None:
    # A single failed building aborts the batch, but the error carries PUMA context
    args = bronze.ComstockPumaTimeseriesRequestArgs(puma_gisjoin=_SAMPLE_PUMA)
    respx.get(args.building_timeseries_url(123)).mock(
        return_value=httpx.Response(404, text="NoSuchKey")
    )
    with httpx.Client() as client:
        with pytest.raises(PipelineError, match=f"PUMA {_SAMPLE_PUMA}"):
            bronze.fetch_puma_timeseries_table(args, client, [123])


# --------------------------------------------------------------------------- #
# Write + manifest + partitioning
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Live integration (real OEDI data lake -- no credentials required)
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_ingest_puma_timeseries_bronze_live(tmp_path: Path) -> None:
    """
    Fetch + shape + write against the real OEDI data lake, three buildings' worth
    """
    args = bronze.ComstockPumaTimeseriesRequestArgs()  # DC, PUMA G11000101
    # ``ingest_puma_timeseries_bronze`` fetches the whole PUMA (~950 requests),
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
        bronze.ComstockTimeseriesBronzeSchema.columns
    )

    bronze.ComstockTimeseriesBronzeSchema.validate(back)
    assert back["bldg_id"].n_unique() == 3
    assert back.height == 3 * 365 * 24 * 4
    assert back["puma_gisjoin"].unique().to_list() == [args.puma_gisjoin]


# --------------------------------------------------------------------------- #
# Per-PUMA metadata (ids + weights in one request)
# --------------------------------------------------------------------------- #


@respx.mock
def test_ingest_puma_metadata_bronze_offline_end_to_end(tmp_path: Path) -> None:
    """One row per building, and the curated columns in schema order."""
    args = bronze.ComstockPumaMetadataRequestArgs(puma_gisjoin=_SAMPLE_PUMA)
    respx.get(args.object_url()).mock(
        return_value=httpx.Response(200, content=_puma_metadata_bytes())
    )

    row = bronze.ingest_puma_metadata_bronze(
        args, root_uri=tmp_path, write_time=_FIXED_WRITE_TIME
    )
    assert row.dataset_name == bronze.PUMA_METADATA_DATASET_NAME

    back = bronze.read_puma_metadata_bronze(args, tmp_path)
    assert back.columns == bronze.PUMA_METADATA_COLUMNS
    # one row per building, unlike the per-county table
    assert back.height == back["bldg_id"].n_unique()
    assert sorted(back["bldg_id"].to_list()) == _sample_puma_bldg_ids()
    assert back["puma_gisjoin"].unique().to_list() == [_SAMPLE_PUMA]


@respx.mock
def test_building_ids_come_from_the_written_bronze(tmp_path: Path) -> None:
    """
    The ids a timeseries fetch needs are a local read once the metadata is on disk.

    Proven by mocking the URL exactly once: a second network resolve would exhaust
    the route and fail.
    """
    args = bronze.ComstockPumaMetadataRequestArgs(puma_gisjoin=_SAMPLE_PUMA)
    route = respx.get(args.object_url()).mock(
        side_effect=[httpx.Response(200, content=_puma_metadata_bytes())]
    )
    bronze.ingest_puma_metadata_bronze(args, root_uri=tmp_path)

    ids = bronze.building_ids(bronze.read_puma_metadata_bronze(args, tmp_path))

    assert route.call_count == 1
    assert ids == _sample_puma_bldg_ids()


def test_the_curated_puma_columns_carry_no_as_simulated_geography() -> None:
    """
    Every geography column in the source file is ``in.as_simulated_*`` -- the
    county, tract and climate zone the model was *simulated* in, which for a DC
    PUMA includes Virginia counties. A column named ``county_gisjoin`` holding
    another state's county is a plausible-looking wrong number, so the location
    stays the PUMA the file is keyed by.
    """
    assert not [c for c in bronze.PUMA_METADATA_RAW_TO_COLUMN if "as_simulated" in c]
    assert "county_gisjoin" not in bronze.PUMA_METADATA_COLUMNS
    assert "tract_gisjoin" not in bronze.PUMA_METADATA_COLUMNS


# --------------------------------------------------------------------------- #
# Weighted PUMA timeseries (the aggregate path)
# --------------------------------------------------------------------------- #

_WEIGHTED_TYPES = ("warehouse", "smalloffice")
