"""
Flow run manifests: one JSON record per run, for provenance.

A run writes ``{root_uri}/_flows/{flow_id}.json`` when it starts
(``status="running"``) and updates it as it goes -- ``input_ids`` once the inputs
are resolved, then a terminal status, an error, and any metadata worth keeping.

Together with the dataset manifest (which stamps each write with the ``flow_id``
that produced it) this answers the lineage questions in both directions: a
dataset's inputs are its flow's ``input_ids``, and a flow's outputs are the
dataset rows carrying its ``flow_id``.
"""

from __future__ import annotations

import datetime
import json
import logging
import traceback
from typing import Any

import duckdb
import pyarrow.fs
import pydantic

import common.models
from common.storage.manifest import _configure_s3, _ensure_local_parent_dir

LOGGER = logging.getLogger(__name__)

FLOW_MANIFEST_DIR: str = "_flows"


class FlowStatus(common.models.CaseInsensitiveStrEnum):
    """Status of a flow run."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FlowManifest(common.models.FrozenModel):
    """One flow run, as stored in its JSON record."""

    flow_id: str = pydantic.Field(description="Prefect flow run UUID")
    flow_name: str = pydantic.Field(description="Prefect flow name")
    writer: str = pydantic.Field(
        description="identity of the user or service that triggered the run"
    )
    scheduled_time: datetime.datetime = pydantic.Field(
        description="UTC timestamp when the flow was submitted (as_of anchor)"
    )
    start_time: datetime.datetime = pydantic.Field(
        description="UTC timestamp when the flow began executing"
    )
    end_time: datetime.datetime | None = pydantic.Field(
        default=None,
        description="UTC timestamp when the flow reached a terminal status",
    )
    status: FlowStatus = pydantic.Field(
        description="running, completed, failed, or cancelled"
    )
    input_ids: list[str] = pydantic.Field(
        default_factory=list,
        description="write_ids of datasets consumed by this flow",
    )
    error: str | None = pydantic.Field(
        default=None, description="traceback if status is failed"
    )
    metadata: dict[str, Any] = pydantic.Field(
        default_factory=dict,
        description="arbitrary key-value bag for intermediate results, metrics, or logs",
    )

    @pydantic.field_validator("metadata", mode="before")
    @classmethod
    def _parse_metadata_json(cls, v: Any) -> dict[str, Any]:
        if isinstance(v, str):
            return json.loads(v)
        return v


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def _flow_manifest_uri(root_uri: str, flow_id: str) -> str:
    return f"{root_uri}/{FLOW_MANIFEST_DIR}/{flow_id}.json"


def _write_flow_manifest(root_uri: str, manifest: FlowManifest) -> None:
    uri = _flow_manifest_uri(root_uri, manifest.flow_id)
    _ensure_local_parent_dir(uri)
    fs, path = pyarrow.fs.FileSystem.from_uri(uri)
    with fs.open_output_stream(path) as f:
        f.write(manifest.model_dump_json().encode())


def _read_flow_manifest_from_storage(root_uri: str, flow_id: str) -> FlowManifest:
    uri = _flow_manifest_uri(root_uri, flow_id)
    fs, path = pyarrow.fs.FileSystem.from_uri(uri)
    try:
        with fs.open_input_stream(path) as f:
            raw = f.read()
    except FileNotFoundError:
        raise KeyError(f"no flow manifest found for flow_id={flow_id!r}") from None
    return FlowManifest.model_validate_json(raw)


_TERMINAL_STATUSES = frozenset(
    {FlowStatus.COMPLETED, FlowStatus.FAILED, FlowStatus.CANCELLED}
)


def write_flow_manifest_start(
    *,
    flow_id: str,
    flow_name: str,
    writer: str,
    root_uri: str,
    scheduled_time: datetime.datetime,
    start_time: datetime.datetime,
) -> FlowManifest:
    """Write the initial record for a run (``status="running"``)."""
    manifest = FlowManifest(
        flow_id=flow_id,
        flow_name=flow_name,
        writer=writer,
        scheduled_time=scheduled_time,
        start_time=start_time,
        status=FlowStatus.RUNNING,
    )
    _write_flow_manifest(root_uri, manifest)
    LOGGER.info("flow manifest started: flow_id=%s flow_name=%s", flow_id, flow_name)
    return manifest


def update_flow_manifest(
    *,
    flow_id: str,
    root_uri: str,
    status: FlowStatus | None = None,
    input_ids: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    error: BaseException | None = None,
) -> FlowManifest:
    """Read-modify-write the record for *flow_id*.

    Only the fields passed change. ``metadata`` is **merged** into what is already
    there, so successive updates can each add their own keys. A terminal *status*
    (completed, failed, cancelled) also stamps ``end_time``.
    """
    current = _read_flow_manifest_from_storage(root_uri, flow_id)

    updates: dict[str, Any] = {}
    if status is not None:
        updates["status"] = status
        if status in _TERMINAL_STATUSES:
            updates["end_time"] = datetime.datetime.now(tz=datetime.UTC)
    if input_ids is not None:
        updates["input_ids"] = input_ids
    if metadata is not None:
        updates["metadata"] = {**current.metadata, **metadata}
    if error is not None:
        updates["error"] = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )

    manifest = current.model_copy(update=updates)
    _write_flow_manifest(root_uri, manifest)
    LOGGER.info("flow manifest updated: flow_id=%s status=%s", flow_id, manifest.status)
    return manifest


# ---------------------------------------------------------------------------
# Read / Query
# ---------------------------------------------------------------------------


_FLOW_COLUMNS_SQL = """{
    flow_id: 'VARCHAR',
    flow_name: 'VARCHAR',
    writer: 'VARCHAR',
    scheduled_time: 'TIMESTAMPTZ',
    start_time: 'TIMESTAMPTZ',
    end_time: 'TIMESTAMPTZ',
    status: 'VARCHAR',
    input_ids: 'VARCHAR[]',
    error: 'VARCHAR',
    metadata: 'JSON'
}"""


def _build_flow_source(root_uri: str) -> str | None:
    """The DuckDB ``FROM`` clause over every flow record, or ``None`` if there are
    none yet."""
    flow_dir = f"{root_uri}/{FLOW_MANIFEST_DIR}"
    fs, dir_path = pyarrow.fs.FileSystem.from_uri(flow_dir)
    file_infos = fs.get_file_info(
        pyarrow.fs.FileSelector(dir_path, allow_not_found=True)
    )
    has_json = any(
        fi.type == pyarrow.fs.FileType.File and fi.path.endswith(".json")
        for fi in file_infos
    )
    if not has_json:
        return None
    glob = f"{flow_dir.rstrip('/')}/*.json".replace("'", "''")
    return f"read_json('{glob}', columns={_FLOW_COLUMNS_SQL}, format='auto')"


def _execute(root_uri: str, query: str, params: list[Any]) -> list[FlowManifest]:
    con = duckdb.connect()
    try:
        if root_uri.startswith("s3://"):
            _configure_s3(con)
        result = con.execute(query, params)
        columns = [desc[0] for desc in result.description]
        records = result.fetchall()
    finally:
        con.close()
    return [FlowManifest.model_validate(dict(zip(columns, r))) for r in records]


def read_flow_manifest(*, flow_id: str, root_uri: str) -> FlowManifest:
    """The record for one run.

    Raises:
        KeyError: If no record exists for *flow_id*.
    """
    source = _build_flow_source(root_uri)
    if source is None:
        raise KeyError(f"no flow manifest found for flow_id={flow_id!r}")
    rows = _execute(root_uri, f"SELECT * FROM {source} WHERE flow_id = $1", [flow_id])
    if not rows:
        raise KeyError(f"no flow manifest found for flow_id={flow_id!r}")
    return rows[0]


def query_flow_manifests(
    *,
    root_uri: str,
    status: FlowStatus | None = None,
    flow_name: str | None = None,
    writer: str | None = None,
    input_id: str | None = None,
) -> list[FlowManifest]:
    """Flow records matching every given filter, most recently started first.

    ``input_id`` finds the runs that consumed a particular dataset ``write_id``.
    """
    source = _build_flow_source(root_uri)
    if source is None:
        return []

    clauses: list[str] = []
    params: list[Any] = []
    for column, value in (
        ("status", status),
        ("flow_name", flow_name),
        ("writer", writer),
    ):
        if value is not None:
            clauses.append(f"{column} = ${len(params) + 1}")
            params.append(value)
    if input_id is not None:
        clauses.append(f"list_contains(input_ids, ${len(params) + 1})")
        params.append(input_id)

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return _execute(
        root_uri, f"SELECT * FROM {source} {where_sql} ORDER BY start_time DESC", params
    )
