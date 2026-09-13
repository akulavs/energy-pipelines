"""
Tests for the climate_pipeline NSRDB bronze ingestion.

Mirrors tests/nsrdb, but exercises the schema-wired module: the joined
``NsrdbPointSliceKey`` (point + dates + interval) is the manifest key and
``NsrdbRequestArgs`` is the fetch config (interval/attributes). Uses the same
real fixtures (a captured NSRDB CSV + the golden bronze table it parses to).
"""

from __future__ import annotations

import datetime as dt
import os
from typing import Literal
from pathlib import Path

import httpx
import polars as pl
import pydantic
import pytest
import respx
from polars.testing import assert_frame_equal

from common.exceptions import PipelineValueError
from common.storage import columnar
from common.storage.manifest import ManifestRow
from external_data.climate_pipeline import failures, schema
from external_data.climate_pipeline.nsrdb import bronze


class _LegacyNsrdbKey(schema.PointSliceKey):
    """The NSRDB key as it was before it recorded the attributes -- point, range
    and interval. Entries in this shape are still on disk."""

    interval: Literal[30, 60]


FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_CSV = FIXTURES / "nsrdb_goes_aggregated_sample.csv"
BRONZE_PARQUET = FIXTURES / "nsrdb_goes_aggregated_bronze.parquet"

# Sample CSV is a real NSRDB download truncated to the first ~100 KB.
_FIXTURE_ROWS = 1274
_FIXED_WRITE_TIME = dt.datetime(2026, 7, 8, tzinfo=dt.timezone.utc)
_NODE = (37.77, -122.42)  # the golden's 4 km cell

_HAS_CREDENTIALS = bool(os.environ.get("NSRDB_API_KEY")) or bronze._rc_path().exists()


@pytest.fixture(scope="session")
def sample_csv() -> str:
    return SAMPLE_CSV.read_text()


@pytest.fixture(scope="session")
def bronze_golden() -> pl.DataFrame:
    return pl.read_parquet(BRONZE_PARQUET)


def _fixture_request(
    start: dt.date = dt.date(2023, 1, 1), end: dt.date = dt.date(2023, 12, 31)
) -> schema.ClimatePipelineRequestArgs:
    """Joined request over 2023 (the golden's year) for the golden's point."""
    return schema.ClimatePipelineRequestArgs(
        points=(_NODE,), start_date=start, end_date=end
    )


# --------------------------------------------------------------------------- #
# Request / query / schema
# --------------------------------------------------------------------------- #


def test_build_query_shape() -> None:
    nsrdb = schema.NsrdbRequestArgs()
    query = bronze.build_query(nsrdb, 37.7749, -122.4194, 2023)
    assert query["wkt"] == "POINT(-122.4194 37.7749)"
    assert query["names"] == "2023"
    assert query["interval"] == str(nsrdb.interval)
    assert query["utc"] == "true"
    assert set(query) >= {"wkt", "attributes", "names", "interval", "utc", "leap_day"}
    # credentials are injected later by download_csv, never in the query
    assert "api_key" not in query and "email" not in query


def test_nsrdb_fetch_config_defaults() -> None:
    nsrdb = schema.NsrdbRequestArgs()
    assert nsrdb.source_dataset == bronze.DATASET
    assert nsrdb.interval in (30, 60)
    assert len(nsrdb.attributes) == len(bronze.VALUE_COLUMNS) == 14


def test_invalid_interval_rejected() -> None:
    with pytest.raises(pydantic.ValidationError, match="should be 30 or 60"):
        schema.NsrdbRequestArgs(interval=15)  # ty:ignore[invalid-argument-type]


def test_attributes_subset_accepted() -> None:
    nsrdb = schema.NsrdbRequestArgs(attributes=("ghi", "dni"))
    assert nsrdb.attributes == ("ghi", "dni")


def test_attributes_normalised_to_curated_order_and_deduped() -> None:
    # Two spellings of one selection must be one request, or the same data
    # refetches under a different manifest key.
    jumbled = schema.NsrdbRequestArgs(attributes=("dni", "ghi", "dni"))
    assert jumbled.attributes == ("ghi", "dni")
    assert jumbled == schema.NsrdbRequestArgs(attributes=("ghi", "dni"))


def test_unknown_attribute_rejected() -> None:
    with pytest.raises(pydantic.ValidationError, match="Unknown NSRDB attribute"):
        schema.NsrdbRequestArgs(attributes=("ghi", "aerosol_optical_depth"))


def test_empty_attributes_rejected() -> None:
    with pytest.raises(pydantic.ValidationError):
        schema.NsrdbRequestArgs(attributes=())


def test_selected_columns_are_the_subset_in_stored_order() -> None:
    nsrdb = schema.NsrdbRequestArgs(attributes=("dni", "ghi"))
    assert bronze.selected_columns(nsrdb) == ["ghi", "dni"]
    assert bronze.selected_columns(None) == bronze.VALUE_COLUMNS
    assert bronze.selected_columns(schema.NsrdbRequestArgs()) == bronze.VALUE_COLUMNS


def test_query_asks_nsrdb_for_only_the_selection() -> None:
    nsrdb = schema.NsrdbRequestArgs(attributes=("ghi", "air_temperature"))
    query = bronze.build_query(nsrdb, 37.77, -122.42, 2023)
    assert query["attributes"] == "ghi,air_temperature"


def test_request_rejects_bad_date_range() -> None:
    with pytest.raises(pydantic.ValidationError, match="on or before"):
        schema.ClimatePipelineRequestArgs(
            points=(_NODE,),
            start_date=dt.date(2023, 2, 1),
            end_date=dt.date(2023, 1, 1),
        )


# --------------------------------------------------------------------------- #
# Fetch
# --------------------------------------------------------------------------- #


@respx.mock
def test_download_csv_hits_endpoint_and_injects_credentials() -> None:
    route = respx.get(f"{bronze.BASE_URL}.csv").mock(
        return_value=httpx.Response(200, text="ok")
    )
    body = bronze.download_csv(
        {"wkt": "POINT(-122.4194 37.7749)", "names": "2023", "interval": "30"},
        api_key="TESTKEY",
        email="analyst@example.com",
    )
    assert body == "ok"
    params = route.calls.last.request.url.params
    assert params["api_key"] == "TESTKEY"
    assert params["email"] == "analyst@example.com"
    assert params["wkt"] == "POINT(-122.4194 37.7749)"
    assert params["names"] == "2023"


# --------------------------------------------------------------------------- #
# Parse
# --------------------------------------------------------------------------- #


def test_bronze_table_matches_golden(
    sample_csv: str, bronze_golden: pl.DataFrame
) -> None:
    table = bronze.csv_to_bronze_table(sample_csv)
    assert_frame_equal(table.sort("valid_time"), bronze_golden.sort("valid_time"))


def test_parse_produces_curated_columns(sample_csv: str) -> None:
    table = bronze.csv_to_bronze_table(sample_csv)
    assert table.columns == bronze.KEY_COLUMNS + bronze.VALUE_COLUMNS
    assert table.height == _FIXTURE_ROWS
    assert not any(table[v].null_count() == table.height for v in bronze.VALUE_COLUMNS)


def test_parse_assembles_valid_time_and_location(sample_csv: str) -> None:
    table = bronze.csv_to_bronze_table(sample_csv)
    first = table.sort("valid_time").row(0, named=True)
    assert first["valid_time"] == dt.datetime(2023, 1, 1, 0, 0, tzinfo=dt.timezone.utc)
    assert first["latitude"] == pytest.approx(_NODE[0])
    assert first["longitude"] == pytest.approx(_NODE[1])
    assert (table["latitude"] == _NODE[0]).all()


def test_parse_fails_loud_on_missing_variable(sample_csv: str) -> None:
    lines = sample_csv.splitlines()
    header = lines[2].split(",")
    idx = header.index("Temperature")
    stripped = lines[:2] + [
        ",".join(v for i, v in enumerate(row.split(",")) if i != idx)
        for row in lines[2:]
    ]
    with pytest.raises(PipelineValueError, match="missing expected variable"):
        bronze.csv_to_bronze_table("\n".join(stripped) + "\n")


def test_parse_reads_a_subset_download(sample_csv: str) -> None:
    # The same CSV, read as a two-attribute fetch: the columns nobody asked for
    # are absent, not missing, so the parse succeeds and projects to them.
    nsrdb = schema.NsrdbRequestArgs(attributes=("ghi", "cloud_type"))
    table = bronze.csv_to_bronze_table(sample_csv, bronze.selected_columns(nsrdb))
    assert table.columns == bronze.KEY_COLUMNS + ["ghi", "cloud_type"]
    assert table.height == _FIXTURE_ROWS
    # The integer-coded column keeps its type through the narrowed parse.
    assert table.schema["cloud_type"] == pl.Int32
    assert table.schema["ghi"] == pl.Float32


def test_parse_still_fails_loud_for_a_selected_attribute(sample_csv: str) -> None:
    # An attribute that *was* asked for and did not arrive is a broken download,
    # whatever the selection.
    lines = sample_csv.splitlines()
    header = lines[2].split(",")
    idx = header.index("Temperature")
    stripped = lines[:2] + [
        ",".join(v for i, v in enumerate(row.split(",")) if i != idx)
        for row in lines[2:]
    ]
    nsrdb = schema.NsrdbRequestArgs(attributes=("ghi", "air_temperature"))
    with pytest.raises(
        PipelineValueError,
        match=r"missing expected variable\(s\) \['air_temperature'\]",
    ):
        bronze.csv_to_bronze_table(
            "\n".join(stripped) + "\n", bronze.selected_columns(nsrdb)
        )


def test_subset_table_is_stored_with_the_full_schema(
    bronze_golden: pl.DataFrame,
) -> None:
    subset = bronze_golden.select(bronze.KEY_COLUMNS + ["ghi", "cloud_type"])

    stored = bronze.stored_shape(subset)

    # Every curated column, the unselected ones null -- and each keeping its own
    # declared type, so the integer-coded columns do not come back as floats.
    assert stored.columns == bronze.KEY_COLUMNS + bronze.VALUE_COLUMNS
    assert stored["ghi"].null_count() == 0
    assert stored["cloud_type"].null_count() == 0
    for column in set(bronze.VALUE_COLUMNS) - {"ghi", "cloud_type"}:
        assert stored[column].null_count() == stored.height
    assert stored.schema["fill_flag"] == pl.Int32
    assert stored.schema["dni"] == pl.Float32
    bronze.NsrdbBronzeSchema.validate(stored)


def test_frame_has_expected_dtypes(sample_csv: str) -> None:
    table = bronze.csv_to_bronze_table(sample_csv)
    bronze.NsrdbBronzeSchema.validate(table)
    assert table.schema["valid_time"] == pl.Datetime("us", "UTC")
    assert table.schema["cloud_type"] == pl.Int32
    assert table.schema["fill_flag"] == pl.Int32
    assert table.schema["ghi"] == pl.Float32


# --------------------------------------------------------------------------- #
# ingest_bronze (one entry per point; the HTTP seam mocked so it runs offline)
# --------------------------------------------------------------------------- #


def _fake_point_year_fetch(golden: pl.DataFrame):
    """Stand-in for ``_fetch_point_year_table`` returning the golden stamped at
    the requested point and rebuilt in the requested year."""

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


def test_ingest_combines_points(
    bronze_golden: pl.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        bronze, "_fetch_point_year_table", _fake_point_year_fetch(bronze_golden)
    )
    request = schema.ClimatePipelineRequestArgs(
        points=((37.77, -122.42), (37.72, -122.44)),
        start_date=dt.date(2023, 1, 1),
        end_date=dt.date(2023, 12, 31),
    )

    rows = bronze.ingest_bronze(
        request, root_uri=tmp_path, write_time=_FIXED_WRITE_TIME
    )

    # One entry per point, each keyed to exactly the point it holds.
    assert len(rows) == 2
    keyed = [
        schema.NsrdbPointSliceKey.model_validate_json(r.params_json).point for r in rows
    ]
    assert set(keyed) == {(37.77, -122.42), (37.72, -122.44)}

    # Single-file writes: no location partition directories.
    assert not list((tmp_path / bronze.DATASET_NAME).glob("*/latitude=*"))
    manifests = list((tmp_path / "_manifests" / bronze.DATASET_NAME).glob("*.json"))
    assert len(manifests) == 2

    back = bronze.read_points_bronze(request, schema.NsrdbRequestArgs(), tmp_path)
    assert back.height == 2 * _FIXTURE_ROWS
    assert back["latitude"].n_unique() == 2


def test_ingest_trims_to_requested_range(
    bronze_golden: pl.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Golden spans Jan 1-27; a Jan 1-2 request must trim the whole-year fetch.
    monkeypatch.setattr(
        bronze, "_fetch_point_year_table", _fake_point_year_fetch(bronze_golden)
    )
    request = _fixture_request(dt.date(2023, 1, 1), dt.date(2023, 1, 2))

    rows = bronze.ingest_bronze(
        request, root_uri=tmp_path, write_time=_FIXED_WRITE_TIME
    )
    assert len(rows) == 1
    back = bronze.read_points_bronze(request, schema.NsrdbRequestArgs(), tmp_path)
    assert 0 < back.height < _FIXTURE_ROWS
    assert (back["valid_time"].dt.date() >= dt.date(2023, 1, 1)).all()
    assert (back["valid_time"].dt.date() <= dt.date(2023, 1, 2)).all()


def test_ingest_fails_loud_when_trim_empties(
    bronze_golden: pl.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A range with no golden rows (golden stops Jan 27) trims to empty -> raise.
    monkeypatch.setattr(
        bronze, "_fetch_point_year_table", _fake_point_year_fetch(bronze_golden)
    )
    request = _fixture_request(dt.date(2023, 6, 1), dt.date(2023, 6, 2))
    with pytest.raises(PipelineValueError, match="no NSRDB rows fell inside"):
        bronze.ingest_bronze(request, root_uri=tmp_path, write_time=_FIXED_WRITE_TIME)


# --------------------------------------------------------------------------- #
# Live integration (network + credentials)
# --------------------------------------------------------------------------- #


def _selecting_point_year_fetch(golden: pl.DataFrame):
    """Stand-in for ``_fetch_point_year_table`` returning only the attributes the
    fetch config asked for, as the real parse does."""
    inner = _fake_point_year_fetch(golden)

    def _outer(latitude, longitude, year, nsrdb, *args, **kwargs) -> pl.DataFrame:
        table = inner(latitude, longitude, year)
        return table.select(bronze.KEY_COLUMNS + bronze.selected_columns(nsrdb))

    return _outer


def test_ingest_runs_on_a_subset_and_nulls_the_rest(
    bronze_golden: pl.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        bronze, "_fetch_point_year_table", _selecting_point_year_fetch(bronze_golden)
    )
    nsrdb = schema.NsrdbRequestArgs(attributes=("ghi", "cloud_type"))
    request = _fixture_request()

    rows = bronze.ingest_bronze(
        request, nsrdb=nsrdb, root_uri=tmp_path, write_time=_FIXED_WRITE_TIME
    )

    # The write records what it holds; the columns it does not hold are null, so
    # the stored shape is the same as a full fetch's.
    assert len(rows) == 1
    key = schema.NsrdbPointSliceKey.model_validate_json(rows[0].params_json)
    assert key.attributes == ("ghi", "cloud_type")
    back = bronze.read_points_bronze(request, nsrdb, tmp_path)
    assert back.columns == bronze.KEY_COLUMNS + bronze.VALUE_COLUMNS
    assert back["ghi"].null_count() == 0
    for column in set(bronze.VALUE_COLUMNS) - {"ghi", "cloud_type"}:
        assert back[column].null_count() == back.height


def test_subset_and_full_writes_read_back_together(
    bronze_golden: pl.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The reason the unselected columns are stored as nulls rather than dropped:
    # one ``pl.read_parquet`` over both files, which needs them to agree.
    monkeypatch.setattr(
        bronze, "_fetch_point_year_table", _selecting_point_year_fetch(bronze_golden)
    )
    narrow_point, wide_point = (37.77, -122.42), (37.72, -122.44)
    day = dt.date(2023, 1, 1)
    bronze.ingest_bronze(
        schema.ClimatePipelineRequestArgs(
            points=(narrow_point,), start_date=day, end_date=day
        ),
        nsrdb=schema.NsrdbRequestArgs(attributes=("ghi",)),
        root_uri=tmp_path,
    )
    bronze.ingest_bronze(
        schema.ClimatePipelineRequestArgs(
            points=(wide_point,), start_date=day, end_date=day
        ),
        root_uri=tmp_path,
    )

    both = bronze.read_points_bronze(
        schema.ClimatePipelineRequestArgs(
            points=(narrow_point, wide_point), start_date=day, end_date=day
        ),
        schema.NsrdbRequestArgs(attributes=("ghi",)),
        tmp_path,
    )
    assert both.columns == bronze.KEY_COLUMNS + bronze.VALUE_COLUMNS
    assert both["latitude"].n_unique() == 2


# --------------------------------------------------------------------------- #
# Coverage: which stored writes answer a request
# --------------------------------------------------------------------------- #


def _entry(
    golden: pl.DataFrame, root: Path, nsrdb: schema.NsrdbRequestArgs
) -> ManifestRow:
    """One written entry for the fixture point, fetched with *nsrdb*."""
    request = _fixture_request()
    return bronze.write_point_bronze(
        golden.select(bronze.KEY_COLUMNS + bronze.selected_columns(nsrdb)),
        _NODE,
        request,
        nsrdb,
        root,
        writer="test",
    )


def test_a_wider_entry_covers_a_narrower_request(
    bronze_golden: pl.DataFrame, tmp_path: Path
) -> None:
    row = _entry(bronze_golden, tmp_path, schema.NsrdbRequestArgs())
    narrower = schema.NsrdbRequestArgs(attributes=("ghi", "dni"))
    assert bronze.coverage_match(narrower)(row)


def test_a_narrower_entry_does_not_cover_a_wider_request(
    bronze_golden: pl.DataFrame, tmp_path: Path
) -> None:
    # The case the recorded selection exists for: without it this entry would
    # read as coverage and the added attributes would never be fetched.
    narrow = schema.NsrdbRequestArgs(attributes=("ghi",))
    row = _entry(bronze_golden, tmp_path, narrow)
    assert not bronze.coverage_match(schema.NsrdbRequestArgs())(row)
    assert bronze.coverage_match(narrow)(row)


def test_the_interval_still_decides_before_the_attributes(
    bronze_golden: pl.DataFrame, tmp_path: Path
) -> None:
    # Both halves of NSRDB's identity in one predicate: a 30-minute write is not
    # coverage for a 60-minute request however many attributes it holds.
    row = _entry(bronze_golden, tmp_path, schema.NsrdbRequestArgs(interval=30))
    assert not bronze.coverage_match(schema.NsrdbRequestArgs(interval=60))(row)
    assert bronze.coverage_match(schema.NsrdbRequestArgs(interval=30))(row)


def test_an_entry_written_before_selections_reads_as_the_curated_set(
    bronze_golden: pl.DataFrame, tmp_path: Path
) -> None:
    # A pre-existing write has no ``attributes`` in its key, and held the whole
    # curated set -- a subset could not be asked for then.
    legacy = columnar.write_dataset(
        bronze_golden,
        bronze.NsrdbBronzeSchema,
        bronze.DATASET_NAME,
        _LegacyNsrdbKey(
            point=_NODE,
            start_date=dt.date(2023, 1, 1),
            end_date=dt.date(2023, 12, 31),
            interval=60,
        ),
        str(tmp_path),
        writer="legacy",
    )
    assert bronze.stored_attributes(legacy) == frozenset(schema.NSRDB_ATTRIBUTES)
    assert bronze.coverage_match(schema.NsrdbRequestArgs())(legacy)


@pytest.mark.integration
@pytest.mark.skipif(
    not _HAS_CREDENTIALS, reason="needs NSRDB credentials for the live API"
)
def test_ingest_live_writes_dataset(tmp_path: Path) -> None:
    """Full pipeline against the real NSRDB API for a short in-coverage window."""
    # Two days at the golden's point. The API is billed per point-year, so a
    # wider range would cost a great deal to prove exactly the same thing.
    request = _fixture_request(dt.date(2023, 1, 1), dt.date(2023, 1, 2))
    bronze.ingest_bronze(request, root_uri=tmp_path)

    key = schema.NsrdbPointSliceKey(
        point=_NODE,
        start_date=request.start_date,
        end_date=request.end_date,
        interval=60,
        attributes=schema.NSRDB_ATTRIBUTES,
    )
    back = columnar.read_dataset(
        bronze.NsrdbBronzeSchema, bronze.DATASET_NAME, key, str(tmp_path)
    ).select(bronze.NsrdbBronzeSchema.columns)

    bronze.NsrdbBronzeSchema.validate(back)
    assert list(back.columns) == bronze.KEY_COLUMNS + bronze.VALUE_COLUMNS
    assert back.height > 0
    # Derived from the request rather than hardcoded, so the point count and the
    # window cannot drift out of step with what was actually fetched.
    assert back["latitude"].n_unique() == len(request.points)
    vt = back["valid_time"]
    assert (vt.dt.date() >= request.start_date).all()
    assert (vt.dt.date() <= request.end_date).all()
    assert not any(back[v].null_count() == back.height for v in bronze.VALUE_COLUMNS)


# --------------------------------------------------------------------------- #
# Per-point read-back
# --------------------------------------------------------------------------- #

_READ_START = dt.date(2023, 1, 1)
_READ_END = dt.date(2023, 1, 27)
_P1 = (37.77, -122.42)
_P2 = (40.72, -73.96)


def _point_request(
    points: tuple[tuple[float, float], ...],
) -> schema.ClimatePipelineRequestArgs:
    return schema.ClimatePipelineRequestArgs(
        points=points, start_date=_READ_START, end_date=_READ_END
    )


def _write_point(
    golden: pl.DataFrame,
    root: Path,
    point: tuple[float, float],
    interval: Literal[30, 60] = schema.NSRDB_SILVER_INTERVAL,
) -> None:
    table = golden.with_columns(
        pl.lit(point[0]).alias("latitude"), pl.lit(point[1]).alias("longitude")
    )
    bronze.write_point_bronze(
        table,
        point,
        _point_request((point,)),
        schema.NsrdbRequestArgs(interval=interval),
        root,
        writer="test",
    )


def _read(
    root: Path,
    points: tuple[tuple[float, float], ...],
    interval: Literal[30, 60] = schema.NSRDB_SILVER_INTERVAL,
) -> pl.DataFrame:
    return bronze.read_points_bronze(
        _point_request(points), schema.NsrdbRequestArgs(interval=interval), root
    )


def test_read_points_bronze_returns_every_requested_point(
    bronze_golden: pl.DataFrame, tmp_path: Path
) -> None:
    for point in (_P1, _P2):
        _write_point(bronze_golden, tmp_path, point)
    assert _read(tmp_path, (_P1, _P2)).height == 2 * bronze_golden.height


def test_read_points_bronze_fails_loud_on_partial_coverage(
    bronze_golden: pl.DataFrame, tmp_path: Path
) -> None:
    # Half a request is not a result: silver would silently lose a location.
    _write_point(bronze_golden, tmp_path, _P1)
    with pytest.raises(PipelineValueError, match="1 of 2 requested point"):
        _read(tmp_path, (_P1, _P2))


def test_read_points_bronze_ignores_another_interval(
    bronze_golden: pl.DataFrame, tmp_path: Path
) -> None:
    # Interval is part of the dataset identity, so a 30-minute write does not
    # satisfy a 60-minute read even for the same point and range.
    _write_point(bronze_golden, tmp_path, _P1, interval=30)
    with pytest.raises(PipelineValueError, match="60-minute interval"):
        _read(tmp_path, (_P1,))


# --------------------------------------------------------------------------- #
# Year list + narrowing a point to the years it actually holds
# --------------------------------------------------------------------------- #


def test_requested_years_spans_the_range() -> None:
    request = _fixture_request(dt.date(2021, 6, 1), dt.date(2024, 2, 1))
    # Whole calendar years, because that is what one request buys.
    assert bronze.requested_years(request) == [2021, 2022, 2023, 2024]


def test_request_units_uses_the_same_year_list() -> None:
    # One unit per HTTP request, so the two must not disagree about the count.
    request = _fixture_request(dt.date(2021, 6, 1), dt.date(2023, 2, 1))
    units = bronze.request_units(request)
    assert len(units) == len(bronze.requested_years(request))


def test_narrow_to_fetched_ends_at_the_last_year_held() -> None:
    # A manifest entry records one range, so a point missing its later years must
    # claim only what it holds -- else covers() reports the hole as covered.
    request = _fixture_request(dt.date(2021, 1, 1), dt.date(2024, 12, 31))
    narrowed = bronze.narrow_to_fetched(request, 2022)
    assert narrowed.end_date == dt.date(2022, 12, 31)
    assert narrowed.start_date == request.start_date
    assert narrowed.points == request.points


def test_narrow_to_fetched_never_widens_the_request() -> None:
    # The last fetched year can be the final, partially-requested one; narrowing
    # must not extend the range past what was asked for.
    request = _fixture_request(dt.date(2023, 1, 1), dt.date(2023, 6, 15))
    assert bronze.narrow_to_fetched(request, 2023).end_date == dt.date(2023, 6, 15)


# --------------------------------------------------------------------------- #
# Sequential ingest: a point-year the provider cannot serve
# --------------------------------------------------------------------------- #


def _fetch_failing_year(golden: pl.DataFrame, dead_year: int, seen: list[int]):
    """Golden for every year but *dead_year*, which is permanently unavailable."""
    good = _fake_point_year_fetch(golden)

    def _inner(latitude, longitude, year, *args, **kwargs) -> pl.DataFrame:
        seen.append(year)
        if year == dead_year:
            raise failures.PermanentFetchError(
                f"NSRDB rejected the request (400): no data for {year}"
            )
        return good(latitude, longitude, year)

    return _inner


def test_ingest_keeps_the_years_it_got_when_a_later_year_is_permanently_gone(
    bronze_golden: pl.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Matches the mapped flow: those requests are already spent, so the fetched
    # years are kept under a key narrowed to what they cover rather than dropped.
    seen: list[int] = []
    monkeypatch.setattr(
        bronze,
        "_fetch_point_year_table",
        _fetch_failing_year(bronze_golden, 2023, seen),
    )
    request = _fixture_request(dt.date(2021, 1, 1), dt.date(2023, 12, 31))

    rows = bronze.ingest_bronze(
        request, root_uri=tmp_path, write_time=_FIXED_WRITE_TIME
    )

    assert len(rows) == 1
    key = schema.NsrdbPointSliceKey.model_validate_json(rows[0].params_json)
    assert key.end_date == dt.date(2022, 12, 31)  # narrowed, not 2023
    assert key.start_date == dt.date(2021, 1, 1)
    assert seen == [2021, 2022, 2023]  # stopped at the dead year


def test_ingest_skips_a_point_whose_first_year_is_permanently_gone(
    bronze_golden: pl.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Nothing salvageable for that point, so there is no honest range to claim and
    # no entry is written -- but it must not take the rest of the batch down.
    dead = _NODE
    other = (40.72, -73.96)
    good = _fake_point_year_fetch(bronze_golden)

    def _inner(latitude, longitude, year, *args, **kwargs) -> pl.DataFrame:
        if (latitude, longitude) == dead:
            raise failures.PermanentFetchError(
                "NSRDB rejected the request (400): no data here"
            )
        return good(latitude, longitude, year)

    monkeypatch.setattr(bronze, "_fetch_point_year_table", _inner)
    request = schema.ClimatePipelineRequestArgs(
        points=(dead, other),
        start_date=dt.date(2021, 1, 1),
        end_date=dt.date(2021, 12, 31),
    )

    rows = bronze.ingest_bronze(
        request, root_uri=tmp_path, write_time=_FIXED_WRITE_TIME
    )

    # Only the live point was written, and at its full requested range.
    assert len(rows) == 1
    key = schema.NsrdbPointSliceKey.model_validate_json(rows[0].params_json)
    assert (key.point,) == (other,)
    assert key.end_date == dt.date(2021, 12, 31)


def test_ingest_still_raises_when_a_year_fails_transiently(
    bronze_golden: pl.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A transient gap is not narrowed: that data is expected to arrive, and
    # recording the short range as final would stop anything looking for the rest.
    good = _fake_point_year_fetch(bronze_golden)

    def _inner(latitude, longitude, year, *args, **kwargs) -> pl.DataFrame:
        if year == 2023:
            raise RuntimeError("connection reset")
        return good(latitude, longitude, year)

    monkeypatch.setattr(bronze, "_fetch_point_year_table", _inner)
    request = _fixture_request(dt.date(2021, 1, 1), dt.date(2023, 12, 31))

    with pytest.raises(RuntimeError, match="connection reset"):
        bronze.ingest_bronze(request, root_uri=tmp_path)


# --------------------------------------------------------------------------- #
# The provider's own view of the request budget
# --------------------------------------------------------------------------- #


def _budget_response(headers: dict[str, str], status: int = 200) -> httpx.Response:
    return httpx.Response(status, text="ok", headers=headers)


def test_observed_budget_reads_the_rate_limit_headers() -> None:
    # The published default is 1,000/hour, but a key may be granted another figure.
    # Reading it is what keeps the assumption checkable instead of load-bearing.
    response = _budget_response(
        {"X-RateLimit-Limit": "1000", "X-RateLimit-Remaining": "742"}
    )
    assert bronze.observed_rate_budget(response) == (1000, 742)


def test_observed_budget_is_none_when_the_server_reports_none() -> None:
    # Absent or unparseable headers must not break a fetch -- the budget is
    # diagnostic, never a precondition.
    assert bronze.observed_rate_budget(_budget_response({})) is None
    assert (
        bronze.observed_rate_budget(
            _budget_response(
                {"X-RateLimit-Limit": "lots", "X-RateLimit-Remaining": "1"}
            )
        )
        is None
    )


@respx.mock
def test_download_warns_when_the_budget_is_nearly_spent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Warn while there is still room to react, rather than once the 429s start.
    respx.get(f"{bronze.BASE_URL}.csv").mock(
        return_value=_budget_response(
            {"X-RateLimit-Limit": "1000", "X-RateLimit-Remaining": "20"}
        )
    )
    with caplog.at_level("WARNING"):
        bronze.download_csv({"names": "2023"}, api_key="K", email="a@example.com")
    assert "budget nearly spent" in caplog.text
    assert "20 of 1000" in caplog.text


@respx.mock
def test_download_is_quiet_when_the_budget_is_healthy(
    caplog: pytest.LogCaptureFixture,
) -> None:
    respx.get(f"{bronze.BASE_URL}.csv").mock(
        return_value=_budget_response(
            {"X-RateLimit-Limit": "1000", "X-RateLimit-Remaining": "900"}
        )
    )
    with caplog.at_level("WARNING"):
        bronze.download_csv({"names": "2023"}, api_key="K", email="a@example.com")
    assert "budget nearly spent" not in caplog.text


@respx.mock
def test_download_notes_a_budget_that_differs_from_the_published_default(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A key granted a different quota is exactly the case a hard-coded assumption
    # would get wrong, so say so rather than sizing backfills against the default.
    respx.get(f"{bronze.BASE_URL}.csv").mock(
        return_value=_budget_response(
            {"X-RateLimit-Limit": "5000", "X-RateLimit-Remaining": "4900"}
        )
    )
    with caplog.at_level("INFO"):
        bronze.download_csv({"names": "2023"}, api_key="K", email="a@example.com")
    assert "5000 requests/hour" in caplog.text
    assert str(bronze.DOCUMENTED_REQUESTS_PER_HOUR) in caplog.text


# --------------------------------------------------------------------------- #
# A 4xx that is NSRDB's backend, not our request
# --------------------------------------------------------------------------- #

_BACKEND_400 = '{"status":400,"errors":["Data processing failure."]}'
_OCEAN_400 = (
    '{"status":400,"errors":["No data available at the provided location",'
    '"Data processing failure."]}'
)


@respx.mock
def test_a_backend_400_is_retried_and_can_succeed(sample_csv: str) -> None:
    # The point of the fix: reclassifying is not enough, the request has to rejoin
    # the retry loop. Without that the first 400 would fall out of the loop.
    route = respx.get(f"{bronze.BASE_URL}.csv").mock(
        side_effect=[
            httpx.Response(400, text=_BACKEND_400),
            httpx.Response(200, text=sample_csv),
        ]
    )
    body = bronze.download_csv(
        {"names": "2023"}, api_key="K", email="a@example.com", backoff_seconds=0
    )
    assert body == sample_csv
    assert route.call_count == 2


@respx.mock
def test_a_backend_400_that_never_clears_is_transient_not_permanent() -> None:
    # Exhausting the retries must not upgrade it to permanent: a permanent verdict
    # is what dead-letters the point out of every later run.
    respx.get(f"{bronze.BASE_URL}.csv").mock(
        return_value=httpx.Response(400, text=_BACKEND_400)
    )
    with pytest.raises(Exception) as caught:  # noqa: PT011 - type is the assertion
        bronze.download_csv(
            {"names": "2023"}, api_key="K", email="a@example.com", backoff_seconds=0
        )
    assert not failures.is_permanent(caught.value)


@respx.mock
def test_a_real_rejection_still_fails_immediately() -> None:
    # The ocean case must keep its old behaviour: permanent, no retries burnt.
    route = respx.get(f"{bronze.BASE_URL}.csv").mock(
        return_value=httpx.Response(400, text=_OCEAN_400)
    )
    with pytest.raises(failures.PermanentFetchError, match="cannot serve"):
        bronze.download_csv(
            {"names": "2023"}, api_key="K", email="a@example.com", backoff_seconds=0
        )
    assert route.call_count == 1
