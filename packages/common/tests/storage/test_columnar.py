"""Tests for common.storage.columnar — write/read round-trip."""

from __future__ import annotations

import datetime
import pathlib

import patito as pt
import polars as pl
import pydantic
import pytest

from common.frames import BaseDataFrameSchema
from common.storage.columnar import read_dataset, read_parquet, write_dataset
from common.storage.manifest import ManifestRow, resolve_manifest, scan_manifest


class _Params(pydantic.BaseModel, frozen=True):
    scenario_id: str


class _MutableParams(pydantic.BaseModel):
    scenario_id: str


class _TestSchema(BaseDataFrameSchema):
    value: float = pt.Field(dtype=pl.Float64)
    label: str = pt.Field(dtype=pl.Utf8)


_DATASET_NAME = "test_columnar"


@pytest.fixture
def root_uri(tmp_path: pathlib.Path) -> str:
    return str(tmp_path)


class TestWriteAndRead:
    def test_round_trip(self, root_uri: str) -> None:
        df = pl.DataFrame({"value": [1.0, 2.0], "label": ["a", "b"]})
        params = _Params(scenario_id="s1")
        ts = datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC)

        write_dataset(
            df,
            _TestSchema,
            _DATASET_NAME,
            params,
            root_uri,
            writer="test",
            write_time=ts,
        )

        result = read_dataset(_TestSchema, _DATASET_NAME, params, root_uri)
        assert result.shape == df.shape
        assert result["value"].to_list() == [1.0, 2.0]

    def test_returns_manifest_row(self, root_uri: str) -> None:
        df = pl.DataFrame({"value": [1.0], "label": ["a"]})
        params = _Params(scenario_id="s1")
        ts = datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC)

        row = write_dataset(
            df,
            _TestSchema,
            _DATASET_NAME,
            params,
            root_uri,
            write_time=ts,
            writer="test",
        )
        assert isinstance(row, ManifestRow)
        assert row.dataset_name == _DATASET_NAME

    def test_read_missing_raises_key_error(self, root_uri: str) -> None:
        params = _Params(scenario_id="nonexistent")
        with pytest.raises(KeyError, match="no 'test_columnar' found"):
            read_dataset(_TestSchema, _DATASET_NAME, params, root_uri)

    def test_naive_write_time_raises(self, root_uri: str) -> None:
        df = pl.DataFrame({"value": [1.0], "label": ["a"]})
        params = _Params(scenario_id="s1")
        ts = datetime.datetime(2024, 1, 1)  # naive

        with pytest.raises(ValueError, match="timezone-aware"):
            write_dataset(
                df,
                _TestSchema,
                _DATASET_NAME,
                params,
                root_uri,
                write_time=ts,
                writer="test",
            )

    def test_mutable_params_warn(self, root_uri: str) -> None:
        df = pl.DataFrame({"value": [1.0], "label": ["a"]})
        with pytest.warns(UserWarning, match="frozen"):
            write_dataset(
                df,
                _TestSchema,
                _DATASET_NAME,
                _MutableParams(scenario_id="s1"),
                root_uri,
                writer="test",
            )

    def test_flow_id_forwarded(self, root_uri: str) -> None:
        df = pl.DataFrame({"value": [1.0], "label": ["a"]})
        params = _Params(scenario_id="s1")
        ts = datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC)

        row = write_dataset(
            df,
            _TestSchema,
            _DATASET_NAME,
            params,
            root_uri,
            writer="test",
            write_time=ts,
            flow_id="f1",
        )
        assert row.flow_id == "f1"


class TestResolveAndReadSeparately:
    def test_resolve_returns_manifest_row(self, root_uri: str) -> None:
        df = pl.DataFrame({"value": [1.0], "label": ["a"]})
        params = _Params(scenario_id="s1")
        ts = datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC)

        written = write_dataset(
            df,
            _TestSchema,
            _DATASET_NAME,
            params,
            root_uri,
            write_time=ts,
            writer="test",
        )

        row = resolve_manifest(
            dataset_name=_DATASET_NAME, params=params, root_uri=root_uri
        )
        assert isinstance(row, ManifestRow)
        assert row.dataset_name == "test_columnar"
        assert row.write_id == written.write_id

    def test_read_parquet_from_uri(self, root_uri: str) -> None:
        df = pl.DataFrame({"value": [1.0, 2.0], "label": ["a", "b"]})
        params = _Params(scenario_id="s1")
        ts = datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC)

        write_dataset(
            df,
            _TestSchema,
            _DATASET_NAME,
            params,
            root_uri,
            write_time=ts,
            writer="test",
        )

        row = resolve_manifest(
            dataset_name=_DATASET_NAME, params=params, root_uri=root_uri
        )
        result = read_parquet(row.data_uri, _TestSchema)
        assert result.shape == df.shape
        assert result["value"].to_list() == [1.0, 2.0]

    def test_resolve_missing_raises(self, root_uri: str) -> None:
        params = _Params(scenario_id="nonexistent")
        with pytest.raises(KeyError):
            resolve_manifest(
                dataset_name=_DATASET_NAME, params=params, root_uri=root_uri
            )


class TestScanManifestLatest:
    def test_empty_when_no_writes(self, root_uri: str) -> None:
        rows = scan_manifest(dataset_name=_DATASET_NAME, root_uri=root_uri)
        assert rows == []

    def test_latest_is_first_row(self, root_uri: str) -> None:
        df = pl.DataFrame({"value": [1.0], "label": ["a"]})
        t1 = datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC)
        t2 = datetime.datetime(2024, 6, 1, tzinfo=datetime.UTC)

        write_dataset(
            df,
            _TestSchema,
            _DATASET_NAME,
            _Params(scenario_id="old"),
            root_uri,
            writer="test",
            write_time=t1,
        )
        write_dataset(
            df,
            _TestSchema,
            _DATASET_NAME,
            _Params(scenario_id="new"),
            root_uri,
            writer="test",
            write_time=t2,
        )

        rows = scan_manifest(dataset_name=_DATASET_NAME, root_uri=root_uri)
        assert len(rows) == 2
        latest = _Params.model_validate_json(rows[0].params_json)
        assert latest.scenario_id == "new"


class TestReadParquet:
    def test_missing_file_is_a_failed_write(self, root_uri: str) -> None:
        with pytest.raises(RuntimeError, match="failed mid-run"):
            read_parquet(f"{root_uri}/ds/nope.parquet", _TestSchema)

    def test_schema_violation_raises(self, root_uri: str) -> None:
        pl.DataFrame({"value": ["not a float"]}).write_parquet(
            f"{root_uri}/bad.parquet"
        )
        with pytest.raises(Exception):  # noqa: B017 -- patito's own error type
            read_parquet(f"{root_uri}/bad.parquet", _TestSchema)
