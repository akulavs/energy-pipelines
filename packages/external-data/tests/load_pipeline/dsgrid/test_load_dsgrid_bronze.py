"""
Tests for the dsgrid 2018 EFS industrial bronze ingestion pipeline (both sources)

One pipeline covers both source files via ``DsgridRequestArgs.source``. Offline
tests run against real ~85 KB slices committed under ``fixtures/`` (4 counties
across DC + DE, the first 168 hours of the 2012 axis):

- ``industrial_sample.dsg`` -- manufacturing subsectors (3 NAICS subsectors).
- ``industrial_gaps_sample.dsg`` -- non-manufacturing sectors (all 3).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import io
from pathlib import Path

import h5py
import httpx
import numpy as np
import polars as pl
import pydantic
import pytest
import respx
from polars.testing import assert_frame_equal

from common.exceptions import PipelineError, PipelineValueError
from common.storage import columnar
from external_data.load_pipeline import dsg_common
from external_data.load_pipeline.dsgrid import bronze

FIXTURES = Path(__file__).parent / "fixtures"

INDUSTRIAL = bronze.DsgridSource.INDUSTRIAL
GAPS = bronze.DsgridSource.GAPS
SOURCES = [INDUSTRIAL, GAPS]

# The golden tables in both fixtures cover Delaware; fixtures keep the first week.
GOLDEN_STATE = "DE"
_FIXTURE_HOURS = 168
_FIXED_WRITE_TIME = dt.datetime(2026, 7, 21, tzinfo=dt.timezone.utc)


# Per-source fixture files + source-specific expectations.
@dataclasses.dataclass(frozen=True)
class _Case:
    sample: str
    metadata_golden: str
    timeseries_golden: str
    url_suffix: str
    dc_sectors: list[str]


_CASES: dict[bronze.DsgridSource, _Case] = {
    INDUSTRIAL: _Case(
        sample="industrial_sample.dsg",
        metadata_golden="dsgrid_industrial_metadata_bronze.parquet",
        timeseries_golden="dsgrid_industrial_timeseries_bronze.parquet",
        url_suffix="/industrial.dsg",
        dc_sectors=["3121", "3241"],  # subsector 3111 absent (null) from DC
    ),
    GAPS: _Case(
        sample="industrial_gaps_sample.dsg",
        metadata_golden="dsgrid_industrial_gaps_metadata_bronze.parquet",
        timeseries_golden="dsgrid_industrial_gaps_timeseries_bronze.parquet",
        url_suffix="/industrial_gaps.dsg",
        dc_sectors=["23"],  # DC has only Construction of the three sectors
    ),
}


def _sample_bytes(source: bronze.DsgridSource) -> bytes:
    return (FIXTURES / _CASES[source].sample).read_bytes()


def _metadata_golden(source: bronze.DsgridSource) -> pl.DataFrame:
    return pl.read_parquet(FIXTURES / _CASES[source].metadata_golden)


def _timeseries_golden(source: bronze.DsgridSource) -> pl.DataFrame:
    return pl.read_parquet(FIXTURES / _CASES[source].timeseries_golden)


# --------------------------------------------------------------------------- #
# Request / provenance
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("source", SOURCES)
def test_request_defaults_and_url(source: bronze.DsgridSource) -> None:
    args = bronze.DsgridRequestArgs(source=source)
    assert args.state == "DC"
    assert args.source_dataset == bronze.SOURCE_DATASET
    assert args.object_url().endswith(_CASES[source].url_suffix)


def test_industrial_url_is_not_the_gaps_file() -> None:
    url = bronze.DsgridRequestArgs(source=INDUSTRIAL).object_url()
    assert url.endswith("/industrial.dsg")
    assert "industrial_gaps.dsg" not in url


def test_source_defaults_to_industrial() -> None:
    assert bronze.DsgridRequestArgs().source is INDUSTRIAL


@pytest.mark.parametrize("state", ["usa", "D", "district", "de"])
def test_invalid_state_rejected(state: str) -> None:
    with pytest.raises(pydantic.ValidationError):
        bronze.DsgridRequestArgs(state=state)


def test_invalid_source_rejected() -> None:
    with pytest.raises(pydantic.ValidationError):
        bronze.DsgridRequestArgs.model_validate({"source": "residential"})


def test_fips_to_county_gisjoin() -> None:
    assert bronze._fips_to_county_gisjoin("11001") == "G1100010"
    assert bronze._fips_to_county_gisjoin("10001") == "G1000010"


# --------------------------------------------------------------------------- #
# Fetch (download_dsg is shared machinery; exercise once via the industrial url)
# --------------------------------------------------------------------------- #


@respx.mock
def test_download_dsg_hits_expected_endpoint() -> None:
    dsg_bytes = _sample_bytes(INDUSTRIAL)
    url = bronze.DsgridRequestArgs().object_url()
    route = respx.get(url).mock(return_value=httpx.Response(200, content=dsg_bytes))
    assert dsg_common.download_dsg(url) == dsg_bytes
    assert route.called


@respx.mock
def test_download_dsg_4xx_raises() -> None:
    url = bronze.DsgridRequestArgs().object_url()
    respx.get(url).mock(return_value=httpx.Response(404, text="NoSuchKey"))
    with pytest.raises(PipelineValueError, match="OEDI rejected the request"):
        dsg_common.download_dsg(url, max_retries=1)


@respx.mock
def test_download_dsg_retries_5xx_then_succeeds() -> None:
    dsg_bytes = _sample_bytes(INDUSTRIAL)
    url = bronze.DsgridRequestArgs().object_url()
    route = respx.get(url).mock(
        side_effect=[
            httpx.Response(503, text="SlowDown"),
            httpx.Response(200, content=dsg_bytes),
        ]
    )
    assert dsg_common.download_dsg(url, max_retries=3, backoff_seconds=0) == dsg_bytes
    assert route.call_count == 2


@respx.mock
def test_download_dsg_persistent_5xx_raises() -> None:
    url = bronze.DsgridRequestArgs().object_url()
    route = respx.get(url).mock(return_value=httpx.Response(500, text="oops"))
    with pytest.raises(PipelineError, match="after 3 attempts"):
        dsg_common.download_dsg(url, max_retries=3, backoff_seconds=0)
    assert route.call_count == 3


@respx.mock
def test_download_dsg_retries_request_error() -> None:
    dsg_bytes = _sample_bytes(INDUSTRIAL)
    url = bronze.DsgridRequestArgs().object_url()
    route = respx.get(url).mock(
        side_effect=[
            httpx.ConnectError("connection reset"),
            httpx.Response(200, content=dsg_bytes),
        ]
    )
    assert dsg_common.download_dsg(url, max_retries=3, backoff_seconds=0) == dsg_bytes
    assert route.call_count == 2


# --------------------------------------------------------------------------- #
# Reconstruct / parse (parametrized over both sources)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("source", SOURCES)
def test_metadata_matches_golden(source: bronze.DsgridSource) -> None:
    profile = bronze.PROFILES[source]
    table = bronze.reconstruct_metadata_table(
        _sample_bytes(source), GOLDEN_STATE, source
    )
    assert table.columns == list(profile.metadata_columns)
    assert_frame_equal(
        table.sort(profile.metadata_key_columns),
        _metadata_golden(source).sort(profile.metadata_key_columns),
    )


@pytest.mark.parametrize("source", SOURCES)
def test_timeseries_matches_golden(source: bronze.DsgridSource) -> None:
    profile = bronze.PROFILES[source]
    table = bronze.reconstruct_timeseries_table(
        _sample_bytes(source), GOLDEN_STATE, source
    )
    assert table.columns == list(profile.timeseries_columns)
    assert_frame_equal(
        table.sort(profile.timeseries_key_columns),
        _timeseries_golden(source).sort(profile.timeseries_key_columns),
    )


@pytest.mark.parametrize("source", SOURCES)
def test_reconstruct_tables_matches_individual(source: bronze.DsgridSource) -> None:
    data = _sample_bytes(source)
    meta, ts = bronze.reconstruct_tables(data, GOLDEN_STATE, source)
    assert_frame_equal(
        meta, bronze.reconstruct_metadata_table(data, GOLDEN_STATE, source)
    )
    assert_frame_equal(
        ts, bronze.reconstruct_timeseries_table(data, GOLDEN_STATE, source)
    )


def test_industrial_end_uses_sum_to_total() -> None:
    profile = bronze.PROFILES[INDUSTRIAL]
    data = _sample_bytes(INDUSTRIAL)
    meta = bronze.reconstruct_metadata_table(data, GOLDEN_STATE, INDUSTRIAL)
    diff = (
        meta.select(profile.annual_enduse_columns).sum_horizontal()
        - meta["annual_electricity_total_mwh"]
    )
    assert diff.abs().max() == pytest.approx(0.0, abs=1e-3)

    ts = bronze.reconstruct_timeseries_table(data, GOLDEN_STATE, INDUSTRIAL)
    ts_diff = (
        ts.select(profile.timeseries_enduse_columns).sum_horizontal()
        - ts["electricity_total_mwh"]
    )
    assert ts_diff.abs().max() == pytest.approx(0.0, abs=1e-1)


@pytest.mark.parametrize("source", SOURCES)
def test_annual_total_equals_hourly_sum(source: bronze.DsgridSource) -> None:
    profile = bronze.PROFILES[source]
    data = _sample_bytes(source)
    meta = bronze.reconstruct_metadata_table(data, GOLDEN_STATE, source)
    ts = bronze.reconstruct_timeseries_table(data, GOLDEN_STATE, source)
    hourly_sum = ts.group_by("county_gisjoin", profile.sector_column).agg(
        pl.col("electricity_total_mwh").sum().alias("summed")
    )
    joined = meta.join(hourly_sum, on=["county_gisjoin", profile.sector_column])
    diff = (joined["annual_electricity_total_mwh"] - joined["summed"]).abs().max()
    assert diff == pytest.approx(0.0, abs=1.0)


@pytest.mark.parametrize("source", SOURCES)
def test_null_sectors_skipped(source: bronze.DsgridSource) -> None:
    # Counties where a (sub)sector is absent are skipped, not written as zeros.
    profile = bronze.PROFILES[source]
    dc = bronze.reconstruct_metadata_table(_sample_bytes(source), "DC", source)
    assert (
        sorted(dc[profile.sector_column].unique().to_list())
        == _CASES[source].dc_sectors
    )
    assert dc["county_gisjoin"].unique().to_list() == ["G1100010"]


@pytest.mark.parametrize("source", SOURCES)
def test_timeseries_timestamps_are_naive_2012(source: bronze.DsgridSource) -> None:
    ts = bronze.reconstruct_timeseries_table(
        _sample_bytes(source), GOLDEN_STATE, source
    )
    timestamps = ts["timestamp"]
    assert timestamps.dtype == pl.Datetime("us")
    assert timestamps.min() == dt.datetime(2012, 1, 1, 1, 0)
    assert timestamps.n_unique() == _FIXTURE_HOURS


# --------------------------------------------------------------------------- #
# Fail-loud
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("source", SOURCES)
def test_unknown_state_fails_loud(source: bronze.DsgridSource) -> None:
    with pytest.raises(PipelineValueError, match="no counties found for state"):
        bronze.reconstruct_metadata_table(_sample_bytes(source), "ZZ", source)


def test_unrecognized_file_fails_loud() -> None:
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        f.create_group("data")  # missing 'enumerations'
    with pytest.raises(PipelineValueError, match="not a recognized .dsg"):
        bronze.reconstruct_metadata_table(buf.getvalue(), "DC")


def _min_dsg(n_time_enum: int, n_data_hours: int) -> bytes:
    """A minimal 1-county / 1-subsector industrial .dsg with a time-axis mismatch."""
    enduse_ids = bronze.PROFILES[INDUSTRIAL].enduse_ids
    n_enduses = len(enduse_ids)
    enum_dt = [("id", "S64"), ("name", "S128")]
    map_dt = [("idx", "<u4"), ("scale", "<f4")]
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        f.attrs["dsgrid"] = "0.2.0"
        enums = f.create_group("enumerations")
        enums.create_dataset(
            "geography",
            data=np.array([(b"11001", b"District of Columbia, DC")], enum_dt),
        )
        enums.create_dataset(
            "enduse",
            data=np.array([(e.encode(), e.encode()) for e in enduse_ids], enum_dt),
        )
        enums.create_dataset(
            "sector", data=np.array([(b"3111", b"Animal Food")], enum_dt)
        )
        stamps = [f"2012-01-01 {i + 1:02d}:00:00-05:00" for i in range(n_time_enum)]
        enums.create_dataset(
            "time", data=np.array([(t.encode(), t.encode()) for t in stamps], enum_dt)
        )
        g = f.create_group("data").create_group("3111")
        g.create_dataset("data", data=np.ones((1, n_enduses, n_data_hours), dtype="f4"))
        g.create_dataset("geographies", data=np.array([(0, 1.0)], map_dt))
        g.create_dataset(
            "enduses", data=np.array([(i, 1.0) for i in range(n_enduses)], map_dt)
        )
        g.create_dataset(
            "times", data=np.array([(i, 1.0) for i in range(n_data_hours)], map_dt)
        )
    return buf.getvalue()


def test_hour_axis_mismatch_fails_loud() -> None:
    # data block has 3 hours but the time enumeration declares 4
    with pytest.raises(PipelineValueError, match="data block has shape"):
        bronze.reconstruct_timeseries_table(
            _min_dsg(n_time_enum=4, n_data_hours=3), "DC", INDUSTRIAL
        )


def test_unexpected_enduses_fail_loud() -> None:
    # A gaps file whose enduse enumeration isn't the single 'energy_consumption'
    enum_dt = [("id", "S64"), ("name", "S128")]
    map_dt = [("idx", "<u4"), ("scale", "<f4")]
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        f.attrs["dsgrid"] = "0.2.0"
        e = f.create_group("enumerations")
        e.create_dataset("geography", data=np.array([(b"11001", b"X, DC")], enum_dt))
        e.create_dataset("enduse", data=np.array([(b"wrong", b"Wrong")], enum_dt))
        e.create_dataset("sector", data=np.array([(b"23", b"Construction")], enum_dt))
        t = "2012-01-01 01:00:00-05:00"
        e.create_dataset("time", data=np.array([(t.encode(), t.encode())], enum_dt))
        g = f.create_group("data").create_group("23")
        g.create_dataset("data", data=np.ones((1, 1, 1), dtype="f4"))
        g.create_dataset("geographies", data=np.array([(0, 1.0)], map_dt))
        g.create_dataset("enduses", data=np.array([(0, 1.0)], map_dt))
        g.create_dataset("times", data=np.array([(0, 1.0)], map_dt))
    with pytest.raises(PipelineValueError, match="unexpected end uses"):
        bronze.reconstruct_metadata_table(buf.getvalue(), "DC", GAPS)


# --------------------------------------------------------------------------- #
# Dtypes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("source", SOURCES)
def test_frame_has_expected_dtypes(source: bronze.DsgridSource) -> None:
    profile = bronze.PROFILES[source]
    data = _sample_bytes(source)

    meta = bronze.reconstruct_metadata_table(data, GOLDEN_STATE, source)
    profile.metadata_schema.validate(meta)
    assert meta.schema["annual_electricity_total_mwh"] == pl.Float64
    assert meta.schema["county_gisjoin"] == pl.String

    ts = bronze.reconstruct_timeseries_table(data, GOLDEN_STATE, source)
    profile.timeseries_schema.validate(ts)
    assert ts.schema["timestamp"] == pl.Datetime("us")
    assert ts.schema["electricity_total_mwh"] == pl.Float32


# --------------------------------------------------------------------------- #
# Write + manifest + partitioning (offline end-to-end via dsg_bytes)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("source", SOURCES)
def test_ingest_metadata_partitions_and_round_trips(
    source: bronze.DsgridSource, tmp_path: Path
) -> None:
    profile = bronze.PROFILES[source]
    params = bronze.DsgridRequestArgs(source=source, state=GOLDEN_STATE)
    row = bronze.ingest_metadata_bronze(
        params,
        root_uri=tmp_path,
        write_time=_FIXED_WRITE_TIME,
        dsg_bytes=_sample_bytes(source),
    )
    assert row.dataset_name == profile.metadata_dataset_name
    assert row.writer == bronze.DEFAULT_WRITER
    assert row.write_time == _FIXED_WRITE_TIME

    # No partition directories: a write is one state, so a ``state=XX/`` directory
    # held one file and repeated the key. The manifest is the index.
    assert not list(tmp_path.rglob("state=*"))
    assert len(list(tmp_path.rglob(f"{profile.metadata_dataset_name}/*.parquet"))) == 1
    assert len(list(tmp_path.rglob("_manifests/**/*.json"))) == 1

    golden = _metadata_golden(source)
    back = columnar.read_dataset(
        profile.metadata_schema, profile.metadata_dataset_name, params, str(tmp_path)
    )
    assert_frame_equal(
        back.sort(profile.metadata_key_columns).select(golden.columns),
        golden.sort(profile.metadata_key_columns),
        check_row_order=True,
    )


@pytest.mark.parametrize("source", SOURCES)
def test_ingest_timeseries_partitions_and_round_trips(
    source: bronze.DsgridSource, tmp_path: Path
) -> None:
    profile = bronze.PROFILES[source]
    params = bronze.DsgridRequestArgs(source=source, state=GOLDEN_STATE)
    row = bronze.ingest_timeseries_bronze(
        params,
        root_uri=tmp_path,
        write_time=_FIXED_WRITE_TIME,
        dsg_bytes=_sample_bytes(source),
    )
    assert row.dataset_name == profile.timeseries_dataset_name

    # One file, like every other dataset here. It split by county before, which
    # scattered one state across 56-62 files -- ~75 KB each for the gaps source --
    # for a pruning nothing performs: every reader takes the whole table.
    assert not list(tmp_path.rglob("county_gisjoin=*"))
    assert not list(tmp_path.rglob(f"{profile.sector_column}=*"))
    assert (
        len(list(tmp_path.rglob(f"{profile.timeseries_dataset_name}/*.parquet"))) == 1
    )

    golden = _timeseries_golden(source)
    back = columnar.read_dataset(
        profile.timeseries_schema,
        profile.timeseries_dataset_name,
        params,
        str(tmp_path),
    )
    assert_frame_equal(
        back.sort(profile.timeseries_key_columns).select(golden.columns),
        golden.sort(profile.timeseries_key_columns),
        check_row_order=True,
    )


# --------------------------------------------------------------------------- #
# ingest_all (both sources in one call, still separate datasets)
# --------------------------------------------------------------------------- #


@respx.mock
def test_ingest_all_writes_every_source(tmp_path: Path) -> None:
    # Mock each source's .dsg download with its committed sample slice.
    routes = {}
    for source in SOURCES:
        url = bronze.DsgridRequestArgs(source=source).object_url()
        routes[source] = respx.get(url).mock(
            return_value=httpx.Response(200, content=_sample_bytes(source))
        )

    # Reuse one client across sources; ingest_all must not close a supplied one.
    client = httpx.Client(follow_redirects=True)
    rows = bronze.ingest_all(
        state=GOLDEN_STATE,
        root_uri=tmp_path,
        client=client,
        write_time=_FIXED_WRITE_TIME,
    )
    assert not client.is_closed
    client.close()

    # Each .dsg is downloaded exactly once (one parse feeds both its datasets).
    assert all(route.call_count == 1 for route in routes.values())

    # Return contract: (metadata, timeseries) per source, in source order.
    expected_order = [
        name
        for source in SOURCES
        for name in (
            bronze.PROFILES[source].metadata_dataset_name,
            bronze.PROFILES[source].timeseries_dataset_name,
        )
    ]
    assert [row.dataset_name for row in rows] == expected_order
    assert len(list(tmp_path.rglob("_manifests/**/*.json"))) == 4

    # Each source's datasets round-trip and match its golden row counts.
    for source in SOURCES:
        profile = bronze.PROFILES[source]
        params = bronze.DsgridRequestArgs(source=source, state=GOLDEN_STATE)
        meta = columnar.read_dataset(
            profile.metadata_schema,
            profile.metadata_dataset_name,
            params,
            str(tmp_path),
        )
        assert meta.height == _metadata_golden(source).height


# --------------------------------------------------------------------------- #
# Live integration (network; public bucket, no credentials)
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_ingest_live_reconstructs_dc_industrial(tmp_path: Path) -> None:
    """Full pipeline against the real public industrial.dsg for DC."""
    args = bronze.DsgridRequestArgs(source=INDUSTRIAL)  # DC
    data = dsg_common.download_dsg(args.object_url())

    ts = bronze.reconstruct_timeseries_table(data, "DC", INDUSTRIAL)
    assert ts["timestamp"].n_unique() == bronze.N_HOURS  # 8784
    assert ts["timestamp"].min() == dt.datetime(2012, 1, 1, 1, 0)
    assert ts["timestamp"].max() == dt.datetime(2013, 1, 1, 0, 0)

    meta = bronze.reconstruct_metadata_table(data, "DC", INDUSTRIAL)
    assert meta.height == 26  # 26 of 86 subsectors present in DC
    assert meta["county_gisjoin"].unique().to_list() == ["G1100010"]
    # unit sanity (native MWh): DC industrial total ~162 GWh
    total_gwh = float(meta["annual_electricity_total_mwh"].sum()) / 1e3
    assert 100 < total_gwh < 250

    bronze.ingest_metadata_bronze(args, root_uri=tmp_path, dsg_bytes=data)
    back = columnar.read_dataset(
        bronze.PROFILES[INDUSTRIAL].metadata_schema,
        bronze.PROFILES[INDUSTRIAL].metadata_dataset_name,
        args,
        str(tmp_path),
    )
    assert back.height == 26


@pytest.mark.integration
def test_ingest_live_reconstructs_dc_gaps(tmp_path: Path) -> None:
    """Full pipeline against the real public industrial_gaps.dsg for DC."""
    args = bronze.DsgridRequestArgs(source=GAPS)  # DC
    data = dsg_common.download_dsg(args.object_url())

    ts = bronze.reconstruct_timeseries_table(data, "DC", GAPS)
    assert ts["timestamp"].n_unique() == bronze.N_HOURS  # 8784
    assert ts["timestamp"].min() == dt.datetime(2012, 1, 1, 1, 0)
    assert ts["timestamp"].max() == dt.datetime(2013, 1, 1, 0, 0)

    meta = bronze.reconstruct_metadata_table(data, "DC", GAPS)
    # DC (dense urban) has only Construction of the three non-mfg sectors
    assert meta["naics_sector"].to_list() == ["23"]
    assert meta["county_gisjoin"].to_list() == ["G1100010"]
    assert float(meta["annual_electricity_total_mwh"].sum()) > 0

    bronze.ingest_timeseries_bronze(args, root_uri=tmp_path, dsg_bytes=data)
    back = columnar.read_dataset(
        bronze.PROFILES[GAPS].timeseries_schema,
        bronze.PROFILES[GAPS].timeseries_dataset_name,
        args,
        str(tmp_path),
    )
    assert back["naics_sector"].unique().to_list() == ["23"]
