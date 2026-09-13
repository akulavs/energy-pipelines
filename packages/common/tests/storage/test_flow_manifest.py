"""Tests for common.storage.flow_manifest — flow run provenance records."""

from __future__ import annotations

import datetime
import json
import pathlib

import pydantic
import pytest

from common.storage.flow_manifest import (
    FlowManifest,
    FlowStatus,
    read_flow_manifest,
    query_flow_manifests,
    update_flow_manifest,
    write_flow_manifest_start,
)
from common.storage.manifest import query_manifest, write_manifest


@pytest.fixture
def root_uri(tmp_path: pathlib.Path) -> str:
    return str(tmp_path)


def _start_flow(
    root_uri: str, flow_id: str = "f1", flow_name: str = "test_flow"
) -> FlowManifest:
    ts = datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC)
    return write_flow_manifest_start(
        flow_id=flow_id,
        flow_name=flow_name,
        writer="test-user",
        root_uri=root_uri,
        scheduled_time=ts,
        start_time=ts,
    )


class TestWriteFlowManifest:
    def test_start_writes_running_status(self, root_uri: str) -> None:
        manifest = _start_flow(root_uri)

        assert manifest.status == FlowStatus.RUNNING
        assert manifest.flow_id == "f1"
        assert manifest.flow_name == "test_flow"
        assert manifest.input_ids == []
        assert manifest.end_time is None
        assert manifest.error is None

        # Verify file exists on disk
        sidecar = pathlib.Path(root_uri) / "_flows" / "f1.json"
        assert sidecar.exists()
        data = json.loads(sidecar.read_text())
        assert data["status"] == "running"

    def test_start_has_empty_metadata(self, root_uri: str) -> None:
        manifest = _start_flow(root_uri)
        assert manifest.metadata == {}


class TestUpdateFlowManifest:
    def test_update_status_to_completed(self, root_uri: str) -> None:
        _start_flow(root_uri)

        completed = update_flow_manifest(
            flow_id="f1",
            root_uri=root_uri,
            input_ids=["w1", "w2"],
            status=FlowStatus.COMPLETED,
        )

        assert completed.status == FlowStatus.COMPLETED
        assert completed.input_ids == ["w1", "w2"]
        assert completed.end_time is not None
        assert completed.error is None

        # Verify file is overwritten
        sidecar = pathlib.Path(root_uri) / "_flows" / "f1.json"
        data = json.loads(sidecar.read_text())
        assert data["status"] == "completed"
        assert data["input_ids"] == ["w1", "w2"]

    def test_update_with_metadata(self, root_uri: str) -> None:
        _start_flow(root_uri)

        completed = update_flow_manifest(
            flow_id="f1",
            root_uri=root_uri,
            status=FlowStatus.COMPLETED,
            metadata={"solver_status": "optimal", "optimal_cost": 42.5},
        )

        assert completed.metadata == {"solver_status": "optimal", "optimal_cost": 42.5}

        # Verify round-trip through disk
        sidecar = pathlib.Path(root_uri) / "_flows" / "f1.json"
        data = json.loads(sidecar.read_text())
        assert data["metadata"]["solver_status"] == "optimal"
        assert data["metadata"]["optimal_cost"] == 42.5

    def test_metadata_merges_across_updates(self, root_uri: str) -> None:
        _start_flow(root_uri)

        update_flow_manifest(
            flow_id="f1",
            root_uri=root_uri,
            metadata={"step": "inputs_resolved"},
        )
        final = update_flow_manifest(
            flow_id="f1",
            root_uri=root_uri,
            status=FlowStatus.COMPLETED,
            metadata={"solver_status": "optimal"},
        )

        assert final.metadata == {"step": "inputs_resolved", "solver_status": "optimal"}

    def test_failed_completion_captures_error(self, root_uri: str) -> None:
        _start_flow(root_uri)

        try:
            msg = "something broke"
            raise ValueError(msg)
        except ValueError as exc:
            completed = update_flow_manifest(
                flow_id="f1",
                root_uri=root_uri,
                status=FlowStatus.FAILED,
                error=exc,
            )

        assert completed.status == FlowStatus.FAILED
        assert completed.error is not None
        assert "something broke" in completed.error

    def test_mid_run_input_ids_persisted(self, root_uri: str) -> None:
        """input_ids set mid-run survive even without a status change."""
        _start_flow(root_uri)

        updated = update_flow_manifest(
            flow_id="f1",
            root_uri=root_uri,
            input_ids=["w1", "w2"],
        )

        assert updated.status == FlowStatus.RUNNING
        assert updated.input_ids == ["w1", "w2"]
        assert updated.end_time is None

    def test_end_time_only_set_on_terminal_status(self, root_uri: str) -> None:
        _start_flow(root_uri)

        # Non-terminal update: no end_time
        updated = update_flow_manifest(
            flow_id="f1", root_uri=root_uri, input_ids=["w1"]
        )
        assert updated.end_time is None

        # Terminal update: end_time set
        completed = update_flow_manifest(
            flow_id="f1", root_uri=root_uri, status=FlowStatus.COMPLETED
        )
        assert completed.end_time is not None

    def test_cancelled_is_terminal_and_stamps_end_time(self, root_uri: str) -> None:
        """
        A run stopped from outside is a third ending, not a failure and not still
        running. Terminal, so it closes the record like the other two -- a run left
        with no ``end_time`` reads as work still in progress in a process that
        exited.
        """
        _start_flow(root_uri)

        cancelled = update_flow_manifest(
            flow_id="f1",
            root_uri=root_uri,
            status=FlowStatus.CANCELLED,
            metadata={"rejected_pumas": [{"puma_gisjoin": "G11009999"}]},
        )

        assert cancelled.status == FlowStatus.CANCELLED
        assert cancelled.end_time is not None
        # Whatever the run had discovered is kept, which is the reason to record it
        # at all rather than leave the manifest saying "running".
        assert cancelled.metadata["rejected_pumas"] == [{"puma_gisjoin": "G11009999"}]

    def test_update_missing_flow_raises(self, root_uri: str) -> None:
        with pytest.raises(KeyError, match="no flow manifest found"):
            update_flow_manifest(
                flow_id="nonexistent", root_uri=root_uri, status=FlowStatus.COMPLETED
            )

    def test_preserves_immutable_fields(self, root_uri: str) -> None:
        start = _start_flow(root_uri)

        completed = update_flow_manifest(
            flow_id="f1", root_uri=root_uri, status=FlowStatus.COMPLETED
        )

        assert completed.flow_id == start.flow_id
        assert completed.flow_name == start.flow_name
        assert completed.writer == start.writer
        assert completed.scheduled_time == start.scheduled_time
        assert completed.start_time == start.start_time


class TestReadFlowManifest:
    def test_read_metadata_round_trip(self, root_uri: str) -> None:
        _start_flow(root_uri)
        update_flow_manifest(
            flow_id="f1",
            root_uri=root_uri,
            status=FlowStatus.COMPLETED,
            metadata={"iterations": 100, "converged": True},
        )

        result = read_flow_manifest(flow_id="f1", root_uri=root_uri)
        assert result.metadata == {"iterations": 100, "converged": True}

    def test_read_existing(self, root_uri: str) -> None:
        _start_flow(root_uri)

        result = read_flow_manifest(flow_id="f1", root_uri=root_uri)
        assert isinstance(result, FlowManifest)
        assert result.flow_id == "f1"

    def test_read_missing_raises(self, root_uri: str) -> None:
        with pytest.raises(KeyError, match="no flow manifest found"):
            read_flow_manifest(flow_id="nonexistent", root_uri=root_uri)


class TestQueryFlowManifests:
    def test_query_by_status(self, root_uri: str) -> None:
        _start_flow(root_uri, flow_id="f1", flow_name="flow_a")
        update_flow_manifest(
            flow_id="f1", root_uri=root_uri, status=FlowStatus.COMPLETED
        )

        # f2 is still running
        _start_flow(root_uri, flow_id="f2", flow_name="flow_b")

        completed = query_flow_manifests(root_uri=root_uri, status=FlowStatus.COMPLETED)
        assert len(completed) == 1
        assert completed[0].flow_id == "f1"

        running = query_flow_manifests(root_uri=root_uri, status=FlowStatus.RUNNING)
        assert len(running) == 1
        assert running[0].flow_id == "f2"

    def test_query_by_input_id(self, root_uri: str) -> None:
        _start_flow(root_uri)
        update_flow_manifest(
            flow_id="f1",
            root_uri=root_uri,
            input_ids=["w1", "w2"],
            status=FlowStatus.COMPLETED,
        )

        # Find flows that consumed w1
        results = query_flow_manifests(root_uri=root_uri, input_id="w1")
        assert len(results) == 1
        assert results[0].flow_id == "f1"

        # w99 was not consumed
        results = query_flow_manifests(root_uri=root_uri, input_id="w99")
        assert results == []

    def test_query_by_writer(self, root_uri: str) -> None:
        ts = datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC)
        write_flow_manifest_start(
            flow_id="f-alice",
            flow_name="flow_a",
            writer="alice",
            root_uri=root_uri,
            scheduled_time=ts,
            start_time=ts,
        )
        write_flow_manifest_start(
            flow_id="f-bob",
            flow_name="flow_b",
            writer="bob",
            root_uri=root_uri,
            scheduled_time=ts,
            start_time=ts,
        )

        results = query_flow_manifests(root_uri=root_uri, writer="alice")
        assert len(results) == 1
        assert results[0].flow_id == "f-alice"

    def test_query_combined_filters(self, root_uri: str) -> None:
        ts = datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC)
        write_flow_manifest_start(
            flow_id="f1",
            flow_name="flow_a",
            writer="alice",
            root_uri=root_uri,
            scheduled_time=ts,
            start_time=ts,
        )
        write_flow_manifest_start(
            flow_id="f2",
            flow_name="flow_a",
            writer="bob",
            root_uri=root_uri,
            scheduled_time=ts,
            start_time=ts,
        )

        results = query_flow_manifests(
            root_uri=root_uri, writer="alice", flow_name="flow_a"
        )
        assert len(results) == 1
        assert results[0].flow_id == "f1"

    def test_empty_when_no_flows(self, root_uri: str) -> None:
        results = query_flow_manifests(root_uri=root_uri)
        assert results == []


class _Params(pydantic.BaseModel, frozen=True):
    scenario_id: str


class TestLineageQueries:
    """Test the three core provenance queries using the flow + dataset manifests."""

    @pytest.fixture
    def lineage_scenario(self, root_uri: str) -> dict:
        """Simulate a flow that consumes two datasets and produces two outputs.

        Returns a dict with all write_ids and flow_id for assertions.
        """
        ts = datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC)
        flow_id = "flow-lineage-test"

        # Upstream datasets (written by some prior flow or seed)
        input_a = write_manifest(
            write_time=ts,
            data_uri="fleet/a.parquet",
            params=_Params(scenario_id="s1"),
            root_uri=root_uri,
            dataset_name="fleet",
            writer="seed",
        )
        input_b = write_manifest(
            write_time=ts,
            data_uri="demand/b.parquet",
            params=_Params(scenario_id="s1"),
            root_uri=root_uri,
            dataset_name="demand",
            writer="seed",
        )

        # Flow produces two outputs, each linked via flow_id
        output_x = write_manifest(
            write_time=ts,
            data_uri="solver_result/x.json",
            params=_Params(scenario_id="s1"),
            root_uri=root_uri,
            dataset_name="solver_result",
            writer="flow",
            flow_id=flow_id,
        )
        output_y = write_manifest(
            write_time=ts,
            data_uri="schedule/y.parquet",
            params=_Params(scenario_id="s1"),
            root_uri=root_uri,
            dataset_name="schedule",
            writer="flow",
            flow_id=flow_id,
        )

        # Write the flow manifest with input_ids
        _start_flow(root_uri, flow_id=flow_id, flow_name="test_pipeline")
        update_flow_manifest(
            flow_id=flow_id,
            root_uri=root_uri,
            input_ids=[input_a.write_id, input_b.write_id],
            status=FlowStatus.COMPLETED,
        )

        return {
            "flow_id": flow_id,
            "input_a": input_a,
            "input_b": input_b,
            "output_x": output_x,
            "output_y": output_y,
        }

    def test_what_inputs_produced_output_x(
        self, root_uri: str, lineage_scenario: dict
    ) -> None:
        """Given an output write_id, find the inputs that produced it.

        Path: output write_id → ManifestRow.flow_id → FlowManifest.input_ids
        """
        output_x = lineage_scenario["output_x"]

        # Step 1: look up the output's flow_id from the dataset manifest
        assert output_x.flow_id == lineage_scenario["flow_id"]

        # Step 2: read the flow manifest to get input_ids
        flow = read_flow_manifest(flow_id=output_x.flow_id, root_uri=root_uri)
        assert set(flow.input_ids) == {
            lineage_scenario["input_a"].write_id,
            lineage_scenario["input_b"].write_id,
        }

    def test_what_outputs_did_flow_x_produce(
        self, root_uri: str, lineage_scenario: dict
    ) -> None:
        """Given a flow_id, find all datasets it produced.

        Path: flow_id → dataset manifest WHERE flow_id = X
        """
        flow_id = lineage_scenario["flow_id"]

        # Query the dataset manifest for all rows with this flow_id.
        # Outputs span two dataset_names, so we query each.
        solver_rows = query_manifest(
            dataset_name="solver_result",
            root_uri=root_uri,
            flow_id=flow_id,
            latest_per_params=False,
        )
        schedule_rows = query_manifest(
            dataset_name="schedule",
            root_uri=root_uri,
            flow_id=flow_id,
            latest_per_params=False,
        )

        output_ids = {r.write_id for r in solver_rows + schedule_rows}
        assert output_ids == {
            lineage_scenario["output_x"].write_id,
            lineage_scenario["output_y"].write_id,
        }

    def test_what_flows_consumed_dataset_y(
        self, root_uri: str, lineage_scenario: dict
    ) -> None:
        """Given a dataset write_id, find all flows that consumed it.

        Path: write_id → flow manifest WHERE write_id IN input_ids
        """
        input_a_id = lineage_scenario["input_a"].write_id

        flows = query_flow_manifests(root_uri=root_uri, input_id=input_a_id)
        assert len(flows) == 1
        assert flows[0].flow_id == lineage_scenario["flow_id"]

        # An unrelated write_id returns no flows
        flows = query_flow_manifests(root_uri=root_uri, input_id="nonexistent")
        assert flows == []
