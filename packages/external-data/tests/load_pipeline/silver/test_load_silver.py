"""
Tests for the load_pipeline dsgrid industrial silver combine.

The fixtures under ``fixtures/`` are **real** slices of the public OEDI data for
Washington, DC, written in bronze schema:

- ``dsgrid_industrial_timeseries_bronze`` -- 26 real DC manufacturing subsectors,
  hourly, 2012-01-01 01:00 .. 2012-01-08 00:00 (interval ending), i.e. 168 hour
  starts.
- ``dsgrid_industrial_gaps_timeseries_bronze`` -- the single non-manufacturing
  sector DC has (construction), same window.

DC has one county, so the county dimension is thin here; the county behaviour is
exercised with a second county derived from the same real rows.

The building-stock fixtures in this directory are still used by the flow tests,
which write them through the real bronze writers.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pydantic
import pytest

from common.exceptions import PipelineValueError
from external_data.load_pipeline import schema, silver
from external_data.load_pipeline.comstock import bronze as comstock_bronze
from external_data.load_pipeline.dsgrid import bronze as dsgrid_bronze
from external_data.load_pipeline.resstock import bronze as resstock_bronze

FIXTURES = Path(__file__).parent / "fixtures"

_UTC = dt.timezone.utc
_FIXED_WRITE_TIME = dt.datetime(2026, 8, 3, tzinfo=_UTC)

_STATE = "DC"
_PUMA = "G11000101"
_COUNTY = "G1100010"
_OTHER_COUNTY = "G1100030"

_N_HOURS = 168
_FIRST_HOUR = dt.datetime(2012, 1, 1, 0, 0)  # interval START of the first hour

# Real DC values for the first source hour, measured from the fixtures.
_MANUFACTURING_H0 = 10.172464
_GAPS_H0 = 4.913149
_TOTAL_H0 = 15.085614

INDUSTRIAL = dsgrid_bronze.DsgridSource.INDUSTRIAL
GAPS = dsgrid_bronze.DsgridSource.GAPS


def _read(name: str) -> pl.DataFrame:
    return pl.read_parquet(FIXTURES / f"{name}.parquet")


@pytest.fixture(scope="session")
def dsgrid_frames() -> dict[dsgrid_bronze.DsgridSource, pl.DataFrame]:
    """The timeseries half of each source."""
    return {
        source: _read(dsgrid_bronze.PROFILES[source].timeseries_dataset_name)
        for source in dsgrid_bronze.PROFILES
    }


@pytest.fixture(scope="session")
def dsgrid_metadata() -> dict[dsgrid_bronze.DsgridSource, pl.DataFrame]:
    """The metadata half of each source."""
    return {
        source: _read(dsgrid_bronze.PROFILES[source].metadata_dataset_name)
        for source in dsgrid_bronze.PROFILES
    }


def _request(**overrides) -> schema.IndustrialLoadRequestArgs:
    return schema.IndustrialLoadRequestArgs(state=_STATE, **overrides)


@pytest.fixture
def bronze_root(
    tmp_path: Path,
    dsgrid_frames: dict[dsgrid_bronze.DsgridSource, pl.DataFrame],
    dsgrid_metadata: dict[dsgrid_bronze.DsgridSource, pl.DataFrame],
) -> Path:
    """Write both real dsgrid halves through the production bronze writers."""
    root = tmp_path / "bronze"
    for params in silver.dsgrid_requests(_STATE):
        dsgrid_bronze.write_timeseries_bronze(
            dsgrid_frames[params.source], params, root, write_time=_FIXED_WRITE_TIME
        )
        dsgrid_bronze.write_metadata_bronze(
            dsgrid_metadata[params.source], params, root, write_time=_FIXED_WRITE_TIME
        )
    return root


# --------------------------------------------------------------------------- #
# Request args
# --------------------------------------------------------------------------- #


def test_industrial_request_is_keyed_by_state_not_puma() -> None:
    """
    dsgrid publishes per state and covers every county, so a PUMA in the key
    would make two identical tables for two PUMAs in the same state.
    """
    fields = set(schema.IndustrialLoadRequestArgs.model_fields)
    assert fields == {"state", "dsgrid"}
    assert "puma_gisjoin" not in fields


def test_industrial_request_defaults_to_both_sources() -> None:
    assert _request().dsgrid.sources == tuple(dsgrid_bronze.DsgridSource)


def test_dsgrid_requests_take_a_bare_state() -> None:
    keys = silver.dsgrid_requests(_STATE)
    assert [k.source for k in keys] == list(dsgrid_bronze.DsgridSource)
    assert {k.state for k in keys} == {_STATE}


def test_dsgrid_requests_honour_a_source_subset() -> None:
    keys = silver.dsgrid_requests(_STATE, schema.DsgridRequestArgs(sources=(GAPS,)))
    assert [k.source for k in keys] == [GAPS]


# --------------------------------------------------------------------------- #
# Union (both sources stacked by kind)
# --------------------------------------------------------------------------- #


def test_harmonise_renames_the_sector_columns(
    dsgrid_metadata: dict[dsgrid_bronze.DsgridSource, pl.DataFrame],
) -> None:
    """
    The two sources name their sector column differently -- that is precisely what
    stops them stacking until it is harmonised.
    """
    for source in (INDUSTRIAL, GAPS):
        bronze = dsgrid_bronze.PROFILES[source]
        # The sector column is derived from the source's profile, not passed in,
        # so there is no second argument that could disagree with the first.
        out = silver.harmonise(dsgrid_metadata[source], source)
        assert silver.SECTOR_COLUMN in out.columns
        assert bronze.sector_column not in out.columns
        assert out[silver.SOURCE_COLUMN].unique().to_list() == [str(source)]


def test_naics_codes_cannot_collide_between_sources(
    dsgrid_metadata: dict[dsgrid_bronze.DsgridSource, pl.DataFrame],
) -> None:
    """4-digit subsectors vs 2-digit sectors, so stacking them is unambiguous."""
    sub = dsgrid_metadata[INDUSTRIAL]["naics_subsector"].unique().to_list()
    sec = dsgrid_metadata[GAPS]["naics_sector"].unique().to_list()
    assert all(len(c) == 4 for c in sub)
    assert all(len(c) == 2 for c in sec)
    assert not set(sub) & set(sec)


def test_union_fails_loud_when_the_sources_collide(
    dsgrid_metadata: dict[dsgrid_bronze.DsgridSource, pl.DataFrame],
) -> None:
    """
    The disjoint-NAICS invariant is checked, not just documented.

    The test above asserts the two DC fixtures happen not to collide. This one
    forces a collision to prove the union refuses it: without the guard, silver
    would emit duplicate ``(county_gisjoin, naics_code)`` rows and every
    downstream sum would double-count in silence.
    """
    collided = dsgrid_metadata[GAPS].with_columns(
        pl.lit(dsgrid_metadata[INDUSTRIAL]["naics_subsector"][0]).alias("naics_sector"),
        pl.lit(dsgrid_metadata[INDUSTRIAL]["county_gisjoin"][0]).alias(
            "county_gisjoin"
        ),
    )
    with pytest.raises(PipelineValueError, match="duplicate row"):
        silver.union_sources(
            {INDUSTRIAL: dsgrid_metadata[INDUSTRIAL], GAPS: collided},
            silver.DsgridIndustrialMetadataSilverSchema,
            silver.METADATA_KEY_COLUMNS,
        )


def test_source_order_does_not_change_the_dataset_key() -> None:
    """
    The same set of files is the same dataset, however the caller spelled it.

    ``model_dump_json`` preserves tuple order and the manifest matches that JSON
    exactly, so an un-canonicalised order would key a second, byte-identical
    table -- and a reader spelling it the other way would get a cache miss and a
    full rebuild.
    """
    forward = schema.DsgridRequestArgs(sources=(INDUSTRIAL, GAPS))
    reversed_ = schema.DsgridRequestArgs(sources=(GAPS, INDUSTRIAL))
    assert forward.sources == reversed_.sources
    assert forward.model_dump_json() == reversed_.model_dump_json()

    # ...and the derived bronze keys follow, since silver builds them from this.
    assert silver.dsgrid_requests("DC", forward) == silver.dsgrid_requests(
        "DC", reversed_
    )


def test_curated_variables_accept_any_ordering() -> None:
    """
    Order is not part of the curated set: the projection is driven by the bronze
    raw -> column map, so a reordered tuple names exactly the same columns.
    Comparing tuples rejected it with an unusable "missing [], unexpected []".
    """
    shuffled = tuple(reversed(schema.RESSTOCK_METADATA_VARIABLES))
    args = schema.ResstockRequestArgs(metadata_variables=shuffled)
    # Accepted, and normalised back to the curated order so the key is stable.
    assert args.metadata_variables == schema.RESSTOCK_METADATA_VARIABLES
    assert args.model_dump_json() == schema.ResstockRequestArgs().model_dump_json()

    with pytest.raises(pydantic.ValidationError, match="curated set"):
        schema.ResstockRequestArgs(metadata_variables=("not_a_variable",))


def test_metadata_union_stacks_both_sources(
    dsgrid_metadata: dict[dsgrid_bronze.DsgridSource, pl.DataFrame],
) -> None:
    table = silver.union_sources(
        dsgrid_metadata,
        silver.DsgridIndustrialMetadataSilverSchema,
        silver.METADATA_KEY_COLUMNS,
    )
    assert table.columns == silver.DsgridIndustrialMetadataSilverSchema.columns
    # rows add rather than multiply -- a union, not a join
    assert table.height == sum(f.height for f in dsgrid_metadata.values()) == 27
    assert set(table[silver.SOURCE_COLUMN].unique()) == {str(INDUSTRIAL), str(GAPS)}


def test_timeseries_union_stacks_both_sources(
    dsgrid_frames: dict[dsgrid_bronze.DsgridSource, pl.DataFrame],
) -> None:
    table = silver.union_sources(
        dsgrid_frames,
        silver.DsgridIndustrialTimeseriesSilverSchema,
        silver.TIMESERIES_KEY_COLUMNS,
    )
    assert table.height == sum(f.height for f in dsgrid_frames.values())
    assert table["timestamp"].n_unique() == _N_HOURS


def test_end_uses_are_null_on_non_manufacturing_rows(
    dsgrid_frames: dict[dsgrid_bronze.DsgridSource, pl.DataFrame],
) -> None:
    """
    The gaps file has no end-use breakdown, so those columns stay null rather than
    being invented as zero -- but the total is populated throughout.
    """
    table = silver.union_sources(
        dsgrid_frames,
        silver.DsgridIndustrialTimeseriesSilverSchema,
        silver.TIMESERIES_KEY_COLUMNS,
    )
    gaps = table.filter(pl.col(silver.SOURCE_COLUMN) == str(GAPS))
    assert gaps["electricity_machine_drive_mwh"].null_count() == gaps.height
    assert gaps["electricity_total_mwh"].null_count() == 0

    mfg = table.filter(pl.col(silver.SOURCE_COLUMN) == str(INDUSTRIAL))
    assert mfg["electricity_machine_drive_mwh"].null_count() == 0


def test_total_is_recoverable_by_summing_the_stack(
    dsgrid_frames: dict[dsgrid_bronze.DsgridSource, pl.DataFrame],
) -> None:
    """
    The point of keeping electricity_total_mwh populated on every row: summing it
    over a county x hour gives the whole industrial load, both halves together.
    """
    table = silver.union_sources(
        dsgrid_frames,
        silver.DsgridIndustrialTimeseriesSilverSchema,
        silver.TIMESERIES_KEY_COLUMNS,
    )
    first = table.filter(pl.col("timestamp") == dt.datetime(2012, 1, 1, 1, 0))
    assert float(
        first["electricity_total_mwh"].cast(pl.Float64).sum()
    ) == pytest.approx(_TOTAL_H0, abs=1e-6)


def test_union_rejects_an_empty_source_set() -> None:
    with pytest.raises(PipelineValueError, match="no dsgrid sources"):
        silver.union_sources(
            {},
            silver.DsgridIndustrialTimeseriesSilverSchema,
            silver.TIMESERIES_KEY_COLUMNS,
        )


def test_union_keeps_the_native_dsgrid_axis(
    dsgrid_frames: dict[dsgrid_bronze.DsgridSource, pl.DataFrame],
) -> None:
    """A conforming of bronze, not a re-basing of it."""
    table = silver.union_sources(
        dsgrid_frames,
        silver.DsgridIndustrialTimeseriesSilverSchema,
        silver.TIMESERIES_KEY_COLUMNS,
    )
    assert table["timestamp"].dt.year().unique().to_list() == [schema.DSGRID_YEAR]
    hours = table["timestamp"].unique().sort().diff().drop_nulls().unique()
    assert hours.to_list() == [schema.DSGRID_INTERVAL]


# --------------------------------------------------------------------------- #
# Build / write / read
# --------------------------------------------------------------------------- #


def test_build_both_tables(bronze_root: Path) -> None:
    md = silver.build_metadata_table(_request(), bronze_root)
    ts = silver.build_timeseries_table(_request(), bronze_root)
    assert md.columns == silver.DsgridIndustrialMetadataSilverSchema.columns
    assert ts.columns == silver.DsgridIndustrialTimeseriesSilverSchema.columns
    assert md.height == 27
    assert ts["timestamp"].n_unique() == _N_HOURS


def test_build_fails_loud_when_bronze_is_missing(tmp_path: Path) -> None:
    with pytest.raises(PipelineValueError, match="bronze not found"):
        silver.build_metadata_table(_request(), tmp_path / "empty")


def test_ingest_writes_one_table_per_kind(bronze_root: Path, tmp_path: Path) -> None:
    silver_root = tmp_path / "silver"
    rows = silver.ingest_industrial_silver(
        _request(),
        bronze_root=bronze_root,
        silver_root=silver_root,
        write_time=_FIXED_WRITE_TIME,
    )
    assert [r.dataset_name for r in rows] == [
        silver.METADATA_DATASET_NAME,
        silver.TIMESERIES_DATASET_NAME,
    ]

    md = silver.read_metadata_silver(_request(), silver_root=silver_root)
    ts = silver.read_timeseries_silver(_request(), silver_root=silver_root)
    assert md.height == 27
    assert set(ts[silver.SOURCE_COLUMN].unique()) == {str(INDUSTRIAL), str(GAPS)}


def test_a_source_subset_stacks_only_that_source(
    bronze_root: Path, tmp_path: Path
) -> None:
    """``sources`` is part of the key, so a gaps-only stack is a distinct dataset."""
    silver_root = tmp_path / "silver"
    gaps_only = _request(dsgrid=schema.DsgridRequestArgs(sources=(GAPS,)))
    silver.ingest_industrial_silver(
        gaps_only, bronze_root=bronze_root, silver_root=silver_root
    )
    ts = silver.read_timeseries_silver(gaps_only, silver_root=silver_root)
    assert set(ts[silver.SOURCE_COLUMN].unique()) == {str(GAPS)}
    assert ts["electricity_machine_drive_mwh"].null_count() == ts.height


# --------------------------------------------------------------------------- #
# Geography + source fetch configs (still shared with the bronze ingests)
# --------------------------------------------------------------------------- #


def test_geography_derives_state_from_the_puma() -> None:
    g = schema.LoadGeography(puma_gisjoin=_PUMA)
    assert g.state_fips == "11"
    assert g.state == _STATE
    assert g.utc_offset_minutes == -300

    pacific = schema.LoadGeography(puma_gisjoin="G06000101")
    assert pacific.state == "CA"
    assert pacific.utc_offset_minutes == -480


def test_unknown_state_fips_fails_loud() -> None:
    with pytest.raises(PipelineValueError, match="unknown state FIPS"):
        schema.LoadGeography(puma_gisjoin="G99000101").state  # noqa: B018


def test_geographies_take_bare_gisjoin_codes() -> None:
    """
    The common case is a typed-in list of codes, so that is what the model reads.

    Spelling each entry as an object exists for the one PUMA in a state's minority
    time zone; making every run form pay that cost for the exception would be the
    wrong trade.
    """
    geographies = schema.LoadGeographies.of(_PUMA, "G06000101")
    assert geographies.puma_gisjoins == (_PUMA, "G06000101")
    assert [g.utc_offset_minutes for g in geographies.pumas] == [-300, -480]

    # ... including straight off a run form, where nothing is typed: a list of
    # codes, or a single code, stands for the whole model
    assert schema.LoadGeographies.model_validate({"pumas": [_PUMA]}).puma_gisjoins == (
        _PUMA,
    )
    assert schema.LoadGeographies.model_validate([_PUMA]).puma_gisjoins == (_PUMA,)
    assert schema.LoadGeographies.model_validate(_PUMA).puma_gisjoins == (_PUMA,)

    # and an entry that needs the offset override is still spelled out
    explicit = schema.LoadGeographies.model_validate(
        [{"puma_gisjoin": "G16000101", "local_standard_utc_offset_minutes": -480}]
    )
    assert explicit.pumas[0].utc_offset_minutes == -480


def test_geographies_collapse_to_states_for_the_state_published_sources() -> None:
    """
    dsgrid and the industrial silver are keyed by state, so the list has to be able
    to say what its distinct states are -- and which PUMA to carry for each.
    """
    geographies = schema.LoadGeographies.of(_PUMA, "G11000105", "G06000101")
    assert geographies.states == (_STATE, "CA")
    assert [g.puma_gisjoin for g in geographies.one_per_state()] == [
        _PUMA,
        "G06000101",
    ]


def test_a_repeated_puma_is_refused() -> None:
    """
    Ingesting a PUMA twice is harmless -- the second pass reuses the first's writes
    -- which is exactly why it must not pass silently: a repeat in a typed-in list
    is a typo, and a run meant to cover three PUMAs would cover two.
    """
    with pytest.raises(pydantic.ValidationError, match="must be unique"):
        schema.LoadGeographies.of(_PUMA, _PUMA)


def test_a_run_is_bounded_by_the_budget_it_can_be_given() -> None:
    """
    A flow's timeout is fixed at import, so the number of PUMAs one run may hold is
    bounded -- and refused loudly rather than quietly shortened, which is the same
    rule the deleted building cap broke.
    """
    limit = schema.MAX_PUMAS_PER_RUN
    codes = [f"G11{i:06d}" for i in range(limit + 1)]
    assert schema.LoadGeographies.of(*codes[:limit]).puma_gisjoins == tuple(
        codes[:limit]
    )
    with pytest.raises(pydantic.ValidationError):
        schema.LoadGeographies.of(*codes)


def test_bronze_key_derivations_combine_geography_and_config() -> None:
    g = schema.LoadGeography(puma_gisjoin=_PUMA)
    key = silver.resstock_timeseries_request(
        g, schema.ResstockRequestArgs(upgrade=3, release_year=2024)
    )
    assert (key.state, key.puma_gisjoin, key.upgrade, key.release_year) == (
        _STATE,
        _PUMA,
        3,
        2024,
    )
    puma_key = silver.comstock_puma_metadata_request(
        g, schema.ComstockRequestArgs(upgrade=1)
    )
    assert (puma_key.state, puma_key.puma_gisjoin, puma_key.upgrade) == (
        _STATE,
        _PUMA,
        1,
    )


def test_curated_variables_are_visible_on_the_source_configs() -> None:
    resstock = schema.ResstockRequestArgs()
    assert resstock.metadata_variables == schema.RESSTOCK_METADATA_VARIABLES
    assert len(resstock.metadata_variables) == 14
    # ComStock's curated metadata comes from the per-PUMA file, which carries no
    # county, tract or climate zone -- those exist there only as ``as_simulated``.
    assert len(schema.ComstockRequestArgs().metadata_variables) == 14


def test_curated_variables_match_the_bronze_projection() -> None:
    assert schema.RESSTOCK_METADATA_VARIABLES == tuple(
        resstock_bronze.METADATA_RAW_TO_COLUMN
    )
    assert schema.COMSTOCK_TIMESERIES_VARIABLES == tuple(
        comstock_bronze.TIMESERIES_RAW_TO_COLUMN
    )


def test_narrowing_variables_fails_loud() -> None:
    with pytest.raises(pydantic.ValidationError, match="must be exactly"):
        schema.ResstockRequestArgs(metadata_variables=("bldg_id", "weight"))


def test_dsgrid_end_uses_are_fixed_not_a_request_field() -> None:
    assert "end_uses" not in schema.DsgridRequestArgs.model_fields
    assert len(schema.DSGRID_INDUSTRIAL_END_USES) == 12
    assert schema.DSGRID_GAPS_END_USES == ("energy_consumption",)
