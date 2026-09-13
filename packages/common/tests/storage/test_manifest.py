"""Tests for common.storage.manifest -- the JSON sidecar manifest."""

from __future__ import annotations

import datetime
import json
import pathlib

import pydantic
import pytest

from common.storage.manifest import (
    ManifestRow,
    query_manifest,
    resolve_manifest,
    scan_manifest,
    write_manifest,
)


class _Params(pydantic.BaseModel, frozen=True):
    scenario_id: str


class _NestedParams(pydantic.BaseModel, frozen=True):
    scenario_id: str
    config: dict[str, str]


_T1 = datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC)
_T2 = datetime.datetime(2024, 6, 1, tzinfo=datetime.UTC)
_T3 = datetime.datetime(2024, 12, 1, tzinfo=datetime.UTC)


@pytest.fixture
def root_uri(tmp_path: pathlib.Path) -> str:
    return str(tmp_path)


def _write(
    root_uri: str,
    scenario: str = "s1",
    *,
    when: datetime.datetime = _T1,
    dataset: str = "ds",
    flow_id: str = "",
) -> ManifestRow:
    return write_manifest(
        write_time=when,
        data_uri=f"{dataset}/{scenario}.parquet",
        params=_Params(scenario_id=scenario),
        root_uri=root_uri,
        dataset_name=dataset,
        writer="test",
        flow_id=flow_id,
    )


class TestWriteManifest:
    def test_writes_one_sidecar_per_write(self, root_uri: str) -> None:
        row = _write(root_uri)
        sidecar = pathlib.Path(root_uri) / "_manifests" / "ds" / f"{row.write_id}.json"
        assert sidecar.exists()
        data = json.loads(sidecar.read_text())
        assert data["dataset_name"] == "ds"
        assert data["writer"] == "test"
        # Stored relative to root, so the tree can move.
        assert data["data_uri"] == "ds/s1.parquet"

    def test_returned_row_has_absolute_data_uri(self, root_uri: str) -> None:
        row = _write(root_uri)
        assert row.data_uri == f"{root_uri}/ds/s1.parquet"

    def test_each_write_gets_a_fresh_write_id(self, root_uri: str) -> None:
        assert _write(root_uri).write_id != _write(root_uri).write_id

    def test_write_time_defaults_to_now_utc(self, root_uri: str) -> None:
        before = datetime.datetime.now(tz=datetime.UTC)
        row = write_manifest(
            data_uri="ds/x.parquet",
            params=_Params(scenario_id="s1"),
            root_uri=root_uri,
            dataset_name="ds",
            writer="test",
        )
        assert row.write_time.tzinfo is not None
        assert row.write_time >= before

    def test_naive_write_time_raises(self, root_uri: str) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            _write(root_uri, when=datetime.datetime(2024, 1, 1))

    def test_flow_id_is_recorded(self, root_uri: str) -> None:
        assert _write(root_uri, flow_id="f1").flow_id == "f1"


class TestResolveManifest:
    def test_returns_latest_matching_params(self, root_uri: str) -> None:
        _write(root_uri, when=_T1)
        newest = _write(root_uri, when=_T2)
        _write(root_uri, "other", when=_T3)

        row = resolve_manifest(
            dataset_name="ds", params=_Params(scenario_id="s1"), root_uri=root_uri
        )
        assert row.write_id == newest.write_id

    def test_as_of_selects_the_write_current_at_that_instant(
        self, root_uri: str
    ) -> None:
        first = _write(root_uri, when=_T1)
        _write(root_uri, when=_T3)

        row = resolve_manifest(
            dataset_name="ds",
            params=_Params(scenario_id="s1"),
            root_uri=root_uri,
            as_of=_T2,
        )
        assert row.write_id == first.write_id

    def test_as_of_before_any_write_raises(self, root_uri: str) -> None:
        _write(root_uri, when=_T2)
        with pytest.raises(KeyError, match="as of"):
            resolve_manifest(
                dataset_name="ds",
                params=_Params(scenario_id="s1"),
                root_uri=root_uri,
                as_of=_T1,
            )

    def test_params_must_match_exactly(self, root_uri: str) -> None:
        _write(root_uri, "s1")
        with pytest.raises(KeyError, match="no 'ds' found"):
            resolve_manifest(
                dataset_name="ds", params=_Params(scenario_id="s2"), root_uri=root_uri
            )

    def test_missing_dataset_raises(self, root_uri: str) -> None:
        with pytest.raises(KeyError, match="no 'ds' found"):
            resolve_manifest(
                dataset_name="ds", params=_Params(scenario_id="s1"), root_uri=root_uri
            )

    def test_naive_as_of_raises(self, root_uri: str) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            resolve_manifest(
                dataset_name="ds",
                params=_Params(scenario_id="s1"),
                root_uri=root_uri,
                as_of=datetime.datetime(2024, 1, 1),
            )


class TestScanManifest:
    def test_empty_when_nothing_written(self, root_uri: str) -> None:
        assert scan_manifest(dataset_name="ds", root_uri=root_uri) == []

    def test_one_row_per_params_newest_first(self, root_uri: str) -> None:
        _write(root_uri, "a", when=_T1)
        a_new = _write(root_uri, "a", when=_T3)
        b = _write(root_uri, "b", when=_T2)

        rows = scan_manifest(dataset_name="ds", root_uri=root_uri)
        assert [r.write_id for r in rows] == [a_new.write_id, b.write_id]

    def test_as_of_rolls_back_each_params(self, root_uri: str) -> None:
        a_old = _write(root_uri, "a", when=_T1)
        _write(root_uri, "a", when=_T3)
        _write(root_uri, "b", when=_T3)

        rows = scan_manifest(dataset_name="ds", root_uri=root_uri, as_of=_T2)
        assert [r.write_id for r in rows] == [a_old.write_id]

    def test_datasets_are_isolated(self, root_uri: str) -> None:
        _write(root_uri, dataset="ds")
        _write(root_uri, dataset="other")
        rows = scan_manifest(dataset_name="ds", root_uri=root_uri)
        assert [r.dataset_name for r in rows] == ["ds"]


class TestQueryManifest:
    def test_full_history_when_not_deduplicated(self, root_uri: str) -> None:
        _write(root_uri, "a", when=_T1)
        _write(root_uri, "a", when=_T2)
        rows = query_manifest(
            dataset_name="ds", root_uri=root_uri, latest_per_params=False
        )
        assert len(rows) == 2
        assert [r.write_time for r in rows] == [_T2, _T1]

    def test_filter_by_flow_id(self, root_uri: str) -> None:
        mine = _write(root_uri, "a", flow_id="f1")
        _write(root_uri, "b", flow_id="f2")
        rows = query_manifest(dataset_name="ds", root_uri=root_uri, flow_id="f1")
        assert [r.write_id for r in rows] == [mine.write_id]

    def test_filter_on_params_field(self, root_uri: str) -> None:
        _write(root_uri, "a")
        _write(root_uri, "b")
        rows = query_manifest(
            dataset_name="ds", root_uri=root_uri, params_where={"scenario_id": "b"}
        )
        assert len(rows) == 1
        assert _Params.model_validate_json(rows[0].params_json).scenario_id == "b"

    def test_filter_on_nested_params_field(self, root_uri: str) -> None:
        for solver in ("highs", "clarabel"):
            write_manifest(
                write_time=_T1,
                data_uri=f"ds/{solver}.parquet",
                params=_NestedParams(scenario_id="s1", config={"solver": solver}),
                root_uri=root_uri,
                dataset_name="ds",
                writer="test",
            )
        rows = query_manifest(
            dataset_name="ds",
            root_uri=root_uri,
            params_where={"config.solver": "highs"},
        )
        assert len(rows) == 1
        assert "highs" in rows[0].params_json

    def test_empty_when_nothing_written(self, root_uri: str) -> None:
        assert query_manifest(dataset_name="ds", root_uri=root_uri) == []


class TestManifestRow:
    def test_round_trips_through_json(self, root_uri: str) -> None:
        row = _write(root_uri, flow_id="f1")
        again = ManifestRow.model_validate_json(row.model_dump_json())
        assert again == row

    def test_is_frozen(self, root_uri: str) -> None:
        row = _write(root_uri)
        with pytest.raises(pydantic.ValidationError):
            row.writer = "someone-else"  # type: ignore[misc]
