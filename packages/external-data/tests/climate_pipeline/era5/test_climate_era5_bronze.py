"""
Tests for the climate_pipeline ERA5-Land bronze ingestion.

Mirrors tests/era5_land, but exercises the schema-wired module: the joined
``ClimatePipelineRequestArgs`` (points + dates) is the manifest key and
``Era5RequestArgs`` is the fetch config. Uses the same real fixtures (a captured
CDS archive + the golden bronze table it parses to).
"""

from __future__ import annotations

import datetime as dt
import json
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import polars as pl
import pydantic
import pytest
import xarray as xr
from pandas.testing import assert_frame_equal
from polars.testing import assert_frame_equal as assert_pl_frame_equal

from common.exceptions import PipelineValueError
from common.storage import columnar
from common.storage.manifest import ManifestRow
from external_data.climate_pipeline import failures, schema
from external_data.climate_pipeline.era5 import bronze

FIXTURES = Path(__file__).parent / "fixtures"
RAW_ZIP = FIXTURES / "era5_land_ts_raw.zip"
REQUEST_JSON = FIXTURES / "era5_land_ts_request.json"
BRONZE_PARQUET = FIXTURES / "era5_land_ts_bronze.parquet"

# Must match the value used to generate the golden parquet.
FIXED_INGESTED_AT = dt.datetime(2026, 1, 15, tzinfo=dt.timezone.utc)

# The fixture was captured for this single point + day; CDS snapped it to _NODE.
_FIXTURE_POINT = (37.45, -122.25)
_FIXTURE_DATE = dt.date(2025, 6, 1)
_NODE = (37.4, -122.2)

_HAS_CREDENTIALS = (Path.home() / ".cdsapirc").exists()


@pytest.fixture(scope="session")
def raw_zip() -> Path:
    return RAW_ZIP


@pytest.fixture(scope="session")
def request_dict() -> dict:
    return json.loads(REQUEST_JSON.read_text())


@pytest.fixture(scope="session")
def bronze_golden() -> pd.DataFrame:
    return pd.read_parquet(BRONZE_PARQUET)


def _fixture_request() -> schema.ClimatePipelineRequestArgs:
    """The joined request that keys the fixture point + day."""
    return schema.ClimatePipelineRequestArgs(
        points=(_FIXTURE_POINT,), start_date=_FIXTURE_DATE, end_date=_FIXTURE_DATE
    )


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, reset index, and canonicalize datetime resolution for comparison."""
    df = df.sort_values("valid_time").reset_index(drop=True).copy()
    for col in df.columns:
        if isinstance(
            df[col].dtype, pd.DatetimeTZDtype
        ) or pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.as_unit("ns")
    return df


def _parse(raw_zip: Path, request: dict, scratch: Path) -> pd.DataFrame:
    return bronze.netcdf_to_bronze_table(
        bronze.extract_netcdfs(raw_zip, scratch), request
    )


# --------------------------------------------------------------------------- #
# Request + hash + schema
# --------------------------------------------------------------------------- #


def test_request_hash_is_deterministic_and_order_independent() -> None:
    a = {"variable": ["t2m"], "location": {"latitude": 1.0, "longitude": 2.0}}
    b = {"location": {"longitude": 2.0, "latitude": 1.0}, "variable": ["t2m"]}
    h = bronze.request_hash(a)
    assert h == bronze.request_hash(b)
    assert len(h) == 16 and all(c in "0123456789abcdef" for c in h)


def test_build_request_reproduces_fixture_payload(request_dict: dict) -> None:
    # build_request renders the exact CDS payload from the ERA5 fetch config
    # (variables) + the joined request's date range + a point.
    req = bronze.build_request(
        schema.Era5RequestArgs(), _FIXTURE_DATE, _FIXTURE_DATE, *_FIXTURE_POINT
    )
    assert req == request_dict


def test_era5_fetch_config_defaults() -> None:
    era5 = schema.Era5RequestArgs()
    assert era5.source_dataset == bronze.DATASET
    assert len(era5.variables) == len(bronze.VALUE_COLUMNS) == 13


def test_variables_subset_accepted() -> None:
    era5 = schema.Era5RequestArgs(variables=("2m_temperature", "snow_cover"))
    assert era5.variables == ("2m_temperature", "snow_cover")


def test_variables_normalised_to_curated_order_and_deduped() -> None:
    # Two spellings of one selection must be one request, or the same data
    # refetches under a different hash and a different manifest key.
    jumbled = schema.Era5RequestArgs(
        variables=("snow_cover", "2m_temperature", "snow_cover")
    )
    assert jumbled.variables == ("2m_temperature", "snow_cover")
    assert jumbled == schema.Era5RequestArgs(variables=("2m_temperature", "snow_cover"))


def test_unknown_variable_rejected() -> None:
    with pytest.raises(pydantic.ValidationError, match="Unknown ERA5 variable"):
        schema.Era5RequestArgs(variables=("2m_temperature", "sea_surface_temperature"))


def test_empty_variables_rejected() -> None:
    with pytest.raises(pydantic.ValidationError):
        schema.Era5RequestArgs(variables=())


def test_value_columns_cover_the_curated_set() -> None:
    # Two orders, deliberately: the curated set is ordered as CDS is asked for it,
    # the columns as bronze stores them. They must still hold the same variables.
    assert set(bronze.VALUE_COLUMNS) == set(schema.ERA5_VARIABLE_COLUMNS.values())
    assert set(bronze.VALUE_COLUMNS) <= set(bronze.Era5LandBronzeSchema.columns)


def test_selected_columns_are_the_subset_in_stored_order() -> None:
    era5 = schema.Era5RequestArgs(variables=("snow_cover", "2m_temperature"))
    assert bronze.selected_columns(era5) == ["t2m", "snowc"]
    assert bronze.selected_columns(None) == bronze.VALUE_COLUMNS
    assert bronze.selected_columns(schema.Era5RequestArgs()) == bronze.VALUE_COLUMNS


def test_build_request_asks_cds_for_only_the_selection() -> None:
    era5 = schema.Era5RequestArgs(variables=("2m_temperature",))
    req = bronze.build_request(era5, _FIXTURE_DATE, _FIXTURE_DATE, *_FIXTURE_POINT)
    assert req["variable"] == ["2m_temperature"]


def test_request_rejects_bad_date_range() -> None:
    with pytest.raises(pydantic.ValidationError, match="on or before"):
        schema.ClimatePipelineRequestArgs(
            points=(_FIXTURE_POINT,),
            start_date=dt.date(2025, 6, 2),
            end_date=dt.date(2025, 6, 1),
        )


def test_request_rejects_empty_points() -> None:
    with pytest.raises(pydantic.ValidationError):
        schema.ClimatePipelineRequestArgs(points=())


# --------------------------------------------------------------------------- #
# extract_netcdfs
# --------------------------------------------------------------------------- #


def test_extract_netcdfs_returns_all_group_files(raw_zip: Path, tmp_path: Path) -> None:
    paths = bronze.extract_netcdfs(raw_zip, tmp_path)
    assert len(paths) == 7
    assert all(p.exists() and p.suffix == ".nc" for p in paths)


def test_extract_netcdfs_passthrough_bare_netcdf(tmp_path: Path) -> None:
    nc = tmp_path / "bare.nc"
    xr.Dataset({"t2m": ("valid_time", [1.0, 2.0])}).to_netcdf(nc, engine="h5netcdf")
    assert bronze.extract_netcdfs(nc, tmp_path) == [nc]


def test_extract_netcdfs_rejects_zip_without_netcdf(tmp_path: Path) -> None:
    bad = tmp_path / "empty.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("readme.txt", "no netcdf here")
    with pytest.raises(ValueError, match="No NetCDF file"):
        bronze.extract_netcdfs(bad, tmp_path)


# --------------------------------------------------------------------------- #
# netcdf_to_bronze_table
# --------------------------------------------------------------------------- #


def test_bronze_table_matches_golden(
    raw_zip: Path, request_dict: dict, bronze_golden: pd.DataFrame, tmp_path: Path
) -> None:
    table = _parse(raw_zip, request_dict, tmp_path)
    assert_frame_equal(_norm(table), _norm(bronze_golden))


def test_bronze_table_shape_and_columns(
    raw_zip: Path, request_dict: dict, tmp_path: Path
) -> None:
    table = _parse(raw_zip, request_dict, tmp_path)
    assert len(table) == 24
    assert list(table.columns) == bronze.KEY_COLUMNS + bronze.VALUE_COLUMNS
    assert not any(table[v].isna().all() for v in bronze.VALUE_COLUMNS)


def test_bronze_table_raises_on_missing_variable(
    request_dict: dict, tmp_path: Path
) -> None:
    nc = tmp_path / "partial.nc"
    xr.Dataset(
        {"t2m": ("valid_time", [280.0, 281.0])},
        coords={"valid_time": [0, 1]},
    ).to_netcdf(nc, engine="h5netcdf")
    with pytest.raises(PipelineValueError, match="missing"):
        bronze.netcdf_to_bronze_table(nc, request_dict)


def test_bronze_table_parses_a_subset_download(
    raw_zip: Path, request_dict: dict, tmp_path: Path
) -> None:
    # The same archive, read as a two-variable fetch: the columns nobody asked
    # for are absent, not missing, so the parse succeeds and projects to them.
    era5 = schema.Era5RequestArgs(variables=("2m_temperature", "snow_cover"))
    table = bronze.netcdf_to_bronze_table(
        bronze.extract_netcdfs(raw_zip, tmp_path),
        request_dict,
        bronze.selected_columns(era5),
    )
    assert list(table.columns) == bronze.KEY_COLUMNS + ["t2m", "snowc"]
    assert len(table) == 24


def test_bronze_table_still_raises_for_a_selected_variable(tmp_path: Path) -> None:
    # A variable that *was* asked for and did not arrive is a broken download,
    # whatever the selection.
    nc = tmp_path / "partial.nc"
    xr.Dataset(
        {"t2m": ("valid_time", [280.0, 281.0])},
        coords={
            "valid_time": [0, 1],
            "latitude": 37.4,
            "longitude": -122.2,
        },
    ).to_netcdf(nc, engine="h5netcdf")
    era5 = schema.Era5RequestArgs(variables=("2m_temperature", "snow_cover"))
    with pytest.raises(
        PipelineValueError, match=r"missing expected variable\(s\) \['snowc'\]"
    ):
        bronze.netcdf_to_bronze_table(nc, {}, bronze.selected_columns(era5))


def test_subset_table_is_stored_with_the_full_schema(
    bronze_golden: pd.DataFrame,
) -> None:
    # What a subset fetch parses to: only the selected columns.
    subset = bronze_golden[bronze.KEY_COLUMNS + ["t2m", "snowc"]]

    frame = bronze.bronze_table_to_frame(subset)

    # ...and what is stored: every curated column, the unselected ones null. One
    # schema for every write is what lets several points' files be read together.
    assert frame.columns == bronze.KEY_COLUMNS + bronze.VALUE_COLUMNS
    assert frame["t2m"].null_count() == 0
    assert frame["snowc"].null_count() == 0
    for column in set(bronze.VALUE_COLUMNS) - {"t2m", "snowc"}:
        assert frame[column].null_count() == frame.height
        assert frame.schema[column] == pl.Float32


# --------------------------------------------------------------------------- #
# ingest_bronze (batch; CDS fetch seam mocked so it runs offline)
# --------------------------------------------------------------------------- #


def _fake_point_fetch(golden: pd.DataFrame):
    """Stand-in for ``_fetch_point_table`` returning the golden stamped at the
    requested point (snapped to the 0.1 deg node, as CDS would)."""

    def _inner(latitude, longitude, *args, **kwargs) -> pd.DataFrame:
        table = golden.copy()
        table["latitude"] = round(latitude, 1)
        table["longitude"] = round(longitude, 1)
        return table

    return _inner


def test_ingest_combines_points(
    bronze_golden: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bronze, "_fetch_point_table", _fake_point_fetch(bronze_golden))
    request = schema.ClimatePipelineRequestArgs(
        points=((37.4, -122.2), (40.0, -120.0)),
        start_date=_FIXTURE_DATE,
        end_date=_FIXTURE_DATE,
    )

    rows = bronze.ingest_bronze(
        request, root_uri=tmp_path, write_time=FIXED_INGESTED_AT
    )

    # One entry per node, each keyed to exactly the point it holds.
    assert len(rows) == 2
    keyed = [
        schema.Era5PointSliceKey.model_validate_json(r.params_json).point for r in rows
    ]
    assert set(keyed) == {(37.4, -122.2), (40.0, -120.0)}

    # Single-file writes: no location partition directories.
    assert not list((tmp_path / bronze.DATASET_NAME).glob("*/latitude=*"))
    manifests = list((tmp_path / "_manifests" / bronze.DATASET_NAME).glob("*.json"))
    assert len(manifests) == 2

    back = bronze.read_points_bronze(request, tmp_path)
    assert back.height == 2 * 24
    assert set(zip(back["latitude"].to_list(), back["longitude"].to_list())) == {
        (37.4, -122.2),
        (40.0, -120.0),
    }


def test_ingest_dedupes_snapped_points(
    bronze_golden: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fetch = _fake_point_fetch(bronze_golden)
    calls: list[tuple[float, float]] = []

    def counting_fetch(latitude, longitude, *args, **kwargs):
        calls.append((latitude, longitude))
        return fetch(latitude, longitude, *args, **kwargs)

    monkeypatch.setattr(bronze, "_fetch_point_table", counting_fetch)
    # Two nearby points that both snap to the same 0.1 deg node.
    request = schema.ClimatePipelineRequestArgs(
        points=((37.40, -122.20), (37.41, -122.21)),
        start_date=_FIXTURE_DATE,
        end_date=_FIXTURE_DATE,
    )

    rows = bronze.ingest_bronze(
        request, root_uri=tmp_path, write_time=FIXED_INGESTED_AT
    )

    assert len(calls) == 1  # one node -> fetched once
    assert len(rows) == 1  # ...and written once
    # Both requested coordinates resolve through that single write.
    back = bronze.read_points_bronze(request, tmp_path)
    assert back.height == 24


def _selecting_point_fetch(golden: pd.DataFrame):
    """Stand-in for ``_fetch_point_table`` that returns only the variables the
    fetch config asked for, as the real parse does."""

    def _inner(latitude, longitude, era5, *args, **kwargs) -> pd.DataFrame:
        columns = bronze.KEY_COLUMNS + bronze.selected_columns(era5)
        table = golden[columns].copy()
        table["latitude"] = round(latitude, 1)
        table["longitude"] = round(longitude, 1)
        return table

    return _inner


def test_ingest_runs_on_a_subset_and_nulls_the_rest(
    bronze_golden: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        bronze, "_fetch_point_table", _selecting_point_fetch(bronze_golden)
    )
    era5 = schema.Era5RequestArgs(variables=("2m_temperature", "snow_cover"))
    request = _fixture_request()

    rows = bronze.ingest_bronze(
        request, era5=era5, root_uri=tmp_path, write_time=FIXED_INGESTED_AT
    )

    # The write records what it holds; the columns it does not hold are null, so
    # the stored shape is the same as a full fetch's.
    assert len(rows) == 1
    key = schema.Era5PointSliceKey.model_validate_json(rows[0].params_json)
    assert key.variables == ("2m_temperature", "snow_cover")
    back = bronze.read_points_bronze(request, tmp_path)
    assert back.columns == bronze.KEY_COLUMNS + bronze.VALUE_COLUMNS
    assert back["t2m"].null_count() == 0
    for column in set(bronze.VALUE_COLUMNS) - {"t2m", "snowc"}:
        assert back[column].null_count() == back.height


def test_subset_and_full_writes_read_back_together(
    bronze_golden: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The reason the unselected columns are stored as nulls rather than dropped:
    # one ``pl.read_parquet`` over both files, which needs them to agree.
    monkeypatch.setattr(
        bronze, "_fetch_point_table", _selecting_point_fetch(bronze_golden)
    )
    narrow = schema.ClimatePipelineRequestArgs(
        points=((37.4, -122.2),), start_date=_FIXTURE_DATE, end_date=_FIXTURE_DATE
    )
    wide = schema.ClimatePipelineRequestArgs(
        points=((40.0, -120.0),), start_date=_FIXTURE_DATE, end_date=_FIXTURE_DATE
    )
    bronze.ingest_bronze(
        narrow,
        era5=schema.Era5RequestArgs(variables=("2m_temperature",)),
        root_uri=tmp_path,
    )
    bronze.ingest_bronze(wide, root_uri=tmp_path)

    both = bronze.read_points_bronze(
        schema.ClimatePipelineRequestArgs(
            points=(*narrow.points, *wide.points),
            start_date=_FIXTURE_DATE,
            end_date=_FIXTURE_DATE,
        ),
        tmp_path,
    )
    assert both.height == 2 * 24
    assert both.columns == bronze.KEY_COLUMNS + bronze.VALUE_COLUMNS


# --------------------------------------------------------------------------- #
# Coverage: which stored writes answer a request
# --------------------------------------------------------------------------- #


def _entry(
    golden: pd.DataFrame, root: Path, era5: schema.Era5RequestArgs | None
) -> ManifestRow:
    """One written entry for the fixture point, fetched with *era5*."""
    return bronze.write_point_bronze(
        golden, _FIXTURE_POINT, _fixture_request(), root, writer="test", era5=era5
    )


def test_a_wider_entry_covers_a_narrower_request(
    bronze_golden: pd.DataFrame, tmp_path: Path
) -> None:
    row = _entry(bronze_golden, tmp_path, schema.Era5RequestArgs())
    narrower = schema.Era5RequestArgs(variables=("2m_temperature",))
    assert bronze.variables_match(narrower)(row)


def test_a_narrower_entry_does_not_cover_a_wider_request(
    bronze_golden: pd.DataFrame, tmp_path: Path
) -> None:
    # The case the recorded selection exists for: without it this entry would
    # read as coverage and the added variables would never be fetched.
    row = _entry(
        bronze_golden, tmp_path, schema.Era5RequestArgs(variables=("2m_temperature",))
    )
    assert not bronze.variables_match(schema.Era5RequestArgs())(row)
    assert bronze.variables_match(
        schema.Era5RequestArgs(variables=("2m_temperature",))
    )(row)


def test_an_entry_written_before_selections_reads_as_the_curated_set(
    bronze_golden: pd.DataFrame, tmp_path: Path
) -> None:
    # A pre-existing write has no ``variables`` in its key, and held the whole
    # curated set -- a subset could not be asked for then.
    legacy = columnar.write_dataset(
        bronze.bronze_table_to_frame(bronze_golden),
        bronze.Era5LandBronzeSchema,
        bronze.DATASET_NAME,
        schema.PointSliceKey(
            point=_FIXTURE_POINT, start_date=_FIXTURE_DATE, end_date=_FIXTURE_DATE
        ),
        str(tmp_path),
        writer="legacy",
    )
    assert bronze.stored_variables(legacy) == frozenset(schema.ERA5_VARIABLES)
    assert bronze.variables_match(schema.Era5RequestArgs())(legacy)


# --------------------------------------------------------------------------- #
# Live integration (network + credentials: CDS)
# --------------------------------------------------------------------------- #


@pytest.mark.integration
@pytest.mark.skipif(
    not _HAS_CREDENTIALS, reason="needs ~/.cdsapirc for the live CDS API"
)
def test_ingest_live_matches_golden(
    bronze_golden: pd.DataFrame, tmp_path: Path
) -> None:
    """
    Full pipeline against the real service for the fixture's immutable past day;
    the result must equal the committed golden table.
    """
    request = _fixture_request()
    bronze.ingest_bronze(request, root_uri=tmp_path, write_time=FIXED_INGESTED_AT)

    back = columnar.read_dataset(
        bronze.Era5LandBronzeSchema, bronze.DATASET_NAME, request, str(tmp_path)
    ).sort("valid_time")
    expected = (
        bronze.bronze_table_to_frame(bronze_golden)
        .select(bronze.Era5LandBronzeSchema.columns)
        .sort("valid_time")
    )
    assert_pl_frame_equal(back.select(bronze.Era5LandBronzeSchema.columns), expected)
    assert not set(Path(tempfile.gettempdir()).glob("era5_land_bronze_batch_*"))


# --------------------------------------------------------------------------- #
# Per-point read-back
# --------------------------------------------------------------------------- #


def _write_point(golden: pd.DataFrame, root: Path, point: tuple[float, float]) -> None:
    table = golden.assign(latitude=point[0], longitude=point[1])
    request = schema.ClimatePipelineRequestArgs(
        points=(point,), start_date=_FIXTURE_DATE, end_date=_FIXTURE_DATE
    )
    bronze.write_point_bronze(table, point, request, root, writer="test")


def _read(root: Path, points: tuple[tuple[float, float], ...]):
    return bronze.read_points_bronze(
        schema.ClimatePipelineRequestArgs(
            points=points, start_date=_FIXTURE_DATE, end_date=_FIXTURE_DATE
        ),
        root,
    )


def test_read_points_bronze_returns_every_requested_point(
    bronze_golden: pd.DataFrame, tmp_path: Path
) -> None:
    points = ((37.42, -122.23), (40.72, -73.96))
    for point in points:
        _write_point(bronze_golden, tmp_path, point)
    assert _read(tmp_path, points).height == 2 * len(bronze_golden)


def test_read_points_bronze_fails_loud_on_partial_coverage(
    bronze_golden: pd.DataFrame, tmp_path: Path
) -> None:
    # Returning only the points that happen to exist would build a silver table
    # quietly missing locations, which no downstream check would catch.
    _write_point(bronze_golden, tmp_path, (37.42, -122.23))
    with pytest.raises(PipelineValueError, match="1 of 2 requested grid node"):
        _read(tmp_path, ((37.42, -122.23), (40.72, -73.96)))


def test_read_points_bronze_names_the_missing_points(
    bronze_golden: pd.DataFrame, tmp_path: Path
) -> None:
    _write_point(bronze_golden, tmp_path, (37.42, -122.23))
    with pytest.raises(PipelineValueError, match=r"\(40.7, -74\)"):
        _read(tmp_path, ((37.42, -122.23), (40.72, -73.96)))


def test_read_points_bronze_treats_one_node_as_complete(
    bronze_golden: pd.DataFrame, tmp_path: Path
) -> None:
    # Two requested coordinates in one 0.1 deg cell share a single download, so
    # asking for both is fully covered by that one write -- not half-missing.
    point = (37.42, -122.23)
    _write_point(bronze_golden, tmp_path, point)
    assert _read(tmp_path, (point, (37.421, -122.231))).height == len(bronze_golden)


def test_read_points_bronze_ignores_an_uncovered_range(
    bronze_golden: pd.DataFrame, tmp_path: Path
) -> None:
    point = (37.42, -122.23)
    _write_point(bronze_golden, tmp_path, point)
    wider = schema.ClimatePipelineRequestArgs(
        points=(point,),
        start_date=_FIXTURE_DATE - dt.timedelta(days=30),
        end_date=_FIXTURE_DATE,
    )
    with pytest.raises(PipelineValueError, match="ingest the bronze slice"):
        bronze.read_points_bronze(wider, tmp_path)


def test_read_points_bronze_as_of_resolves_the_pre_refresh_version(
    bronze_golden: pd.DataFrame, tmp_path: Path
) -> None:
    # A refresh writes a new version rather than overwriting, so a reader pinned
    # before it still sees the old data -- this is what makes a silver build
    # reproducible after bronze is refreshed underneath it.
    point = (37.42, -122.23)
    request = schema.ClimatePipelineRequestArgs(
        points=(point,), start_date=_FIXTURE_DATE, end_date=_FIXTURE_DATE
    )
    first = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    second = dt.datetime(2026, 6, 1, tzinfo=dt.timezone.utc)
    table = bronze_golden.assign(latitude=point[0], longitude=point[1])
    bronze.write_point_bronze(
        table, point, request, tmp_path, writer="v1", write_time=first
    )
    bronze.write_point_bronze(
        table.head(5), point, request, tmp_path, writer="v2", write_time=second
    )

    between = dt.datetime(2026, 3, 1, tzinfo=dt.timezone.utc)
    assert bronze.read_points_bronze(request, tmp_path, between).height == len(
        bronze_golden
    )
    assert bronze.read_points_bronze(request, tmp_path).height == 5


def test_ingest_skips_a_permanently_unavailable_point(
    bronze_golden: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The sequential path needs the same tolerance as the mapped flow: one point
    # the provider will never serve must not cost the rest of a long backfill.
    dead = (0.0, -140.0)
    inner = _fake_point_fetch(bronze_golden)

    def fetch(latitude, longitude, *args, **kwargs):
        if (latitude, longitude) == dead:
            raise failures.PermanentFetchError("Request has not produced any data.")
        return inner(latitude, longitude, *args, **kwargs)

    monkeypatch.setattr(bronze, "_fetch_point_table", fetch)
    request = schema.ClimatePipelineRequestArgs(
        points=((37.42, -122.23), dead, (40.72, -73.96)),
        start_date=_FIXTURE_DATE,
        end_date=_FIXTURE_DATE,
    )

    rows = bronze.ingest_bronze(request, root_uri=tmp_path)

    assert len(rows) == 2  # the two reachable points still landed


def test_ingest_propagates_a_transient_failure(
    bronze_golden: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Only *permanent* failures are absorbed. There is no retry layer here, so a
    # transient one must stay visible instead of silently dropping a point.
    def fetch(latitude, longitude, *args, **kwargs):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(bronze, "_fetch_point_table", fetch)
    request = schema.ClimatePipelineRequestArgs(
        points=((37.42, -122.23),), start_date=_FIXTURE_DATE, end_date=_FIXTURE_DATE
    )
    with pytest.raises(RuntimeError, match="connection reset"):
        bronze.ingest_bronze(request, root_uri=tmp_path)


def test_ingest_raises_when_every_point_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fetch(latitude, longitude, *args, **kwargs):
        raise failures.PermanentFetchError("Request has not produced any data.")

    monkeypatch.setattr(bronze, "_fetch_point_table", fetch)
    request = schema.ClimatePipelineRequestArgs(
        points=((0.0, -140.0),), start_date=_FIXTURE_DATE, end_date=_FIXTURE_DATE
    )
    with pytest.raises(PipelineValueError, match="permanently unavailable"):
        bronze.ingest_bronze(request, root_uri=tmp_path)


# --------------------------------------------------------------------------- #
# The resolve/read split silver actually runs
# --------------------------------------------------------------------------- #


def test_resolve_point_uris_keys_at_node_grain(
    bronze_golden: pd.DataFrame, tmp_path: Path
) -> None:
    # Silver groups NSRDB points onto ERA5 nodes by looking each one up in *these*
    # keys, so the grain they come back at is load-bearing: exact coordinates here
    # would silently stop every NSRDB point matching its node, and the join would
    # quietly produce ERA5-only rows. read_points_bronze hides this behind a frame.
    requested = (37.421, -122.231)  # a few metres off the stored coordinate
    _write_point(bronze_golden, tmp_path, (37.42, -122.23))

    uris = bronze.resolve_point_uris(
        schema.ClimatePipelineRequestArgs(
            points=(requested,), start_date=_FIXTURE_DATE, end_date=_FIXTURE_DATE
        ),
        tmp_path,
    )

    assert list(uris) == [bronze.node(requested)]
    assert uris[bronze.node(requested)].endswith(".parquet")


def test_resolve_point_uris_returns_one_uri_per_covered_node(
    bronze_golden: pd.DataFrame, tmp_path: Path
) -> None:
    points = ((37.42, -122.23), (40.72, -73.96))
    for point in points:
        _write_point(bronze_golden, tmp_path, point)

    uris = bronze.resolve_point_uris(
        schema.ClimatePipelineRequestArgs(
            points=points, start_date=_FIXTURE_DATE, end_date=_FIXTURE_DATE
        ),
        tmp_path,
    )

    assert set(uris) == {bronze.node(p) for p in points}
    assert len(set(uris.values())) == 2  # a distinct file per node


def test_read_bronze_uris_dedupes_and_sorts(
    bronze_golden: pd.DataFrame, tmp_path: Path
) -> None:
    # The node-by-node silver build reads a list of resolved files directly, so the
    # dedupe/sort that read_points_bronze used to own has to live in this half.
    _write_point(bronze_golden, tmp_path, (37.42, -122.23))
    uris = bronze.resolve_point_uris(
        schema.ClimatePipelineRequestArgs(
            points=((37.42, -122.23),),
            start_date=_FIXTURE_DATE,
            end_date=_FIXTURE_DATE,
        ),
        tmp_path,
    )
    (uri,) = uris.values()

    # The same file twice stands in for two entries holding overlapping rows.
    frame = bronze.read_bronze_uris([uri, uri])

    assert frame.height == bronze.read_bronze_uris([uri]).height
    assert frame.select(bronze.KEY_COLUMNS).n_unique() == frame.height
    assert frame["valid_time"].is_sorted()


# --------------------------------------------------------------------------- #
# A response with timestamps but no measurements
# --------------------------------------------------------------------------- #


def _all_null_like(golden: pd.DataFrame) -> pd.DataFrame:
    """The golden's shape and time axis, with every variable missing -- what CDS
    returns for a point outside ERA5-Land's land domain."""
    table = golden.copy()
    for column in bronze.VALUE_COLUMNS:
        table[column] = pd.NA
    return table


def test_fetch_rejects_a_response_with_no_values(
    bronze_golden: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An ocean point returns the full time axis with nothing behind it. Stored, it
    # would read as covered forever and silver would emit an all-null node -- so it
    # has to fail here, where the response is still in hand.
    monkeypatch.setattr(
        bronze, "_fetch_point_table", lambda *a, **k: _all_null_like(bronze_golden)
    )
    with pytest.raises(failures.PermanentFetchError, match="no values"):
        bronze.fetch_point_table((34.5, -122.0), _fixture_request())


def test_the_no_values_failure_is_permanent(
    bronze_golden: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Permanent, not transient: ERA5-Land's land mask is the same next run, so a
    # retry buys nothing. This is what stops the point burning its retry budget
    # and what gets it dead-lettered instead.
    monkeypatch.setattr(
        bronze, "_fetch_point_table", lambda *a, **k: _all_null_like(bronze_golden)
    )
    try:
        bronze.fetch_point_table((34.5, -122.0), _fixture_request())
    except failures.PermanentFetchError as exc:
        assert failures.is_permanent(exc)
    else:
        pytest.fail("expected a PermanentFetchError")


def test_fetch_accepts_a_partially_populated_response(
    bronze_golden: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A coastal node can be legitimately patchy -- some variables missing, some
    # present. Only a response with *nothing* in it is rejected.
    patchy = _all_null_like(bronze_golden)
    patchy["t2m"] = bronze_golden["t2m"]
    monkeypatch.setattr(bronze, "_fetch_point_table", lambda *a, **k: patchy)

    table = bronze.fetch_point_table((37.42, -122.23), _fixture_request())
    assert table["t2m"].notna().any()


def test_fetch_rejects_an_empty_response(monkeypatch: pytest.MonkeyPatch) -> None:
    # No rows at all is the same failure wearing a different shape.
    empty = pd.DataFrame({c: [] for c in ["valid_time", "latitude", "longitude"]})
    monkeypatch.setattr(bronze, "_fetch_point_table", lambda *a, **k: empty)
    with pytest.raises(failures.PermanentFetchError, match="no values"):
        bronze.fetch_point_table((34.5, -122.0), _fixture_request())
