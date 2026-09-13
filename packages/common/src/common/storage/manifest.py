"""
Dataset manifests: one JSON sidecar per write, queried with DuckDB.

Every dataset write appends a file at
``{root_uri}/_manifests/{dataset_name}/{write_id}.json`` recording what was
written (``params_json``), where (``data_uri``), when (``write_time``), and by
whom (``writer``, ``flow_id``). Nothing is mutated in place, so a read is a scan
of that directory: the newest row whose params match, optionally clamped to writes
at or before ``as_of``.

The manifest is the only index. There is no database and no catalog file to keep
consistent; a dataset's history is the set of sidecars in its directory.
"""

from __future__ import annotations

import datetime
import json
import logging
import uuid
import warnings
from typing import Any

import duckdb
import pyarrow.fs
import pydantic

import common.models

LOGGER = logging.getLogger(__name__)

MANIFEST_DIR: str = "_manifests"


class ManifestRow(common.models.FrozenModel):
    """One dataset write, as stored in its JSON sidecar."""

    write_id: str = pydantic.Field(description="UUID generated per write")
    dataset_name: str = pydantic.Field(
        description="stable name identifying the schema and computation type"
    )
    write_time: datetime.datetime = pydantic.Field(
        description="UTC timestamp when the write completed"
    )
    params_json: str = pydantic.Field(
        description="JSON-serialised frozen Pydantic params model"
    )
    data_uri: str = pydantic.Field(
        description="path relative to root_uri where the dataset is stored"
    )
    writer: str = pydantic.Field(
        description="identity of the user or service that triggered the write"
    )
    flow_id: str = pydantic.Field(default="", description="Prefect flow run UUID")


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def _ensure_local_parent_dir(file_uri: str) -> None:
    """Create the parent directory of *file_uri* if the filesystem is local."""
    fs, path = pyarrow.fs.FileSystem.from_uri(file_uri)
    if isinstance(fs, pyarrow.fs.LocalFileSystem):
        fs.create_dir(path.rsplit("/", 1)[0], recursive=True)


def _warn_if_mutable_params(params: pydantic.BaseModel) -> None:
    """Emit a warning if *params* is not a frozen Pydantic model."""
    if not type(params).model_config.get("frozen"):
        warnings.warn(
            f"params should be a frozen pydantic.BaseModel (frozen=True), "
            f"got {type(params).__name__}. A mutable model may produce "
            f"inconsistent lookup keys.",
            UserWarning,
            stacklevel=3,
        )


def _require_aware(name: str, value: datetime.datetime | None) -> None:
    if value is not None and value.tzinfo is None:
        msg = f"{name} must be timezone-aware (e.g. datetime.datetime.now(tz=datetime.UTC))"
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def write_manifest(
    *,
    data_uri: str,
    params: pydantic.BaseModel,
    root_uri: str,
    dataset_name: str,
    writer: str,
    write_time: datetime.datetime | None = None,
    flow_id: str = "",
) -> ManifestRow:
    """Record one completed dataset write as a JSON sidecar.

    *data_uri* is stored **relative** to *root_uri* so the whole tree can be moved;
    the returned row has it resolved to an absolute URI for the caller's
    convenience.

    *write_time* defaults to now (UTC). Pass one explicitly to give several writes
    from one logical result the same timestamp, so they are atomic for
    point-in-time reads. Must be timezone-aware when given.
    """
    if write_time is None:
        write_time = datetime.datetime.now(tz=datetime.UTC)
    _require_aware("write_time", write_time)

    write_id = str(uuid.uuid4())
    row = ManifestRow(
        write_id=write_id,
        dataset_name=dataset_name,
        write_time=write_time,
        params_json=params.model_dump_json(),
        data_uri=data_uri,
        writer=writer,
        flow_id=flow_id,
    )
    manifest_uri = f"{root_uri}/{MANIFEST_DIR}/{dataset_name}/{write_id}.json"
    _ensure_local_parent_dir(manifest_uri)

    fs, path = pyarrow.fs.FileSystem.from_uri(manifest_uri)
    with fs.open_output_stream(path) as f:
        f.write(row.model_dump_json().encode())

    LOGGER.info(
        "manifest written: dataset=%s write_id=%s writer=%s data_uri=%s",
        dataset_name,
        write_id,
        writer,
        data_uri,
    )
    return row.model_copy(update={"data_uri": f"{root_uri}/{data_uri}"})


# ---------------------------------------------------------------------------
# DuckDB query helpers
# ---------------------------------------------------------------------------


# Column types for read_json, so a partially written or empty sidecar cannot
# change the inferred schema from one scan to the next.
_MANIFEST_COLUMNS_SQL = """{
    write_id: 'VARCHAR',
    dataset_name: 'VARCHAR',
    write_time: 'TIMESTAMPTZ',
    params_json: 'VARCHAR',
    data_uri: 'VARCHAR',
    writer: 'VARCHAR',
    flow_id: 'VARCHAR'
}"""


def _build_manifest_source(root_uri: str, dataset_name: str) -> str | None:
    """The DuckDB ``FROM`` clause over a dataset's sidecars, or ``None`` if it has
    none yet."""
    manifest_dir = f"{root_uri}/{MANIFEST_DIR}/{dataset_name}"
    fs, dir_path = pyarrow.fs.FileSystem.from_uri(manifest_dir)
    file_infos = fs.get_file_info(
        pyarrow.fs.FileSelector(dir_path, allow_not_found=True)
    )
    has_json = any(
        fi.type == pyarrow.fs.FileType.File and fi.path.endswith(".json")
        for fi in file_infos
    )
    if not has_json:
        return None
    glob = f"{manifest_dir.rstrip('/')}/*.json".replace("'", "''")
    return f"read_json('{glob}', columns={_MANIFEST_COLUMNS_SQL}, format='auto')"


def _configure_s3(con: duckdb.DuckDBPyConnection) -> None:
    """Load the httpfs extension for S3 access. No-op if unavailable."""
    try:
        con.execute("LOAD httpfs")
    except duckdb.IOException, duckdb.CatalogException:
        pass


def _execute_manifest_query(
    root_uri: str,
    source: str,
    where_clauses: list[str],
    params: list[Any],
    *,
    qualify: str = "",
    order_by: str = "",
    limit: int | None = None,
) -> list[ManifestRow]:
    """Run one query over *source* and hydrate the rows.

    I/O errors (permission denied, corrupt files) propagate. The "no sidecars at
    all" case is the caller's, via :func:`_build_manifest_source`.
    """
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    limit_sql = f"LIMIT {limit}" if limit is not None else ""
    query = f"SELECT * FROM {source} {where_sql} {qualify} {order_by} {limit_sql}"
    LOGGER.debug("manifest query: %s params=%s", query, params)

    con = duckdb.connect()
    try:
        if root_uri.startswith("s3://"):
            _configure_s3(con)
        result = con.execute(query, params)
        columns = [desc[0] for desc in result.description]
        records = result.fetchall()
    finally:
        con.close()

    rows: list[ManifestRow] = []
    for record in records:
        d = dict(zip(columns, record))
        d["data_uri"] = f"{root_uri}/{d['data_uri']}"
        rows.append(ManifestRow.model_validate(d))
    return rows


_LATEST_PER_PARAMS = (
    "QUALIFY ROW_NUMBER() OVER (PARTITION BY params_json ORDER BY write_time DESC) = 1"
)


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def query_manifest(
    *,
    dataset_name: str,
    root_uri: str,
    as_of: datetime.datetime | None = None,
    flow_id: str | None = None,
    params_where: dict[str, Any] | None = None,
    latest_per_params: bool = True,
) -> list[ManifestRow]:
    """Rows of *dataset_name*'s manifest, newest first.

    ``as_of`` keeps only writes at or before that instant (timezone-aware).
    ``flow_id`` keeps only the writes one flow run produced.
    ``params_where`` filters on fields *inside* ``params_json`` by equality, with
    dot notation for nesting: ``{"request.year": 2018}`` becomes
    ``json_extract(params_json, '$.request.year') = 2018``.
    ``latest_per_params`` (the default) collapses the history to the newest write
    per distinct ``params_json``; ``False`` returns every write.
    """
    _require_aware("as_of", as_of)

    clauses: list[str] = []
    params: list[Any] = []
    if as_of is not None:
        clauses.append(f"write_time <= ${len(params) + 1}")
        params.append(as_of)
    if flow_id is not None:
        clauses.append(f"flow_id = ${len(params) + 1}")
        params.append(flow_id)
    for key, val in (params_where or {}).items():
        clauses.append(
            f"json_extract(params_json::JSON, ${len(params) + 1}) = ${len(params) + 2}::JSON"
        )
        params.extend(["$." + key, json.dumps(val)])

    source = _build_manifest_source(root_uri, dataset_name)
    if source is None:
        return []
    return _execute_manifest_query(
        root_uri,
        source,
        clauses,
        params,
        qualify=_LATEST_PER_PARAMS if latest_per_params else "",
        order_by="ORDER BY write_time DESC",
    )


def scan_manifest(
    *,
    dataset_name: str,
    root_uri: str,
    as_of: datetime.datetime | None = None,
) -> list[ManifestRow]:
    """The newest write for each distinct ``params_json`` of *dataset_name*,
    optionally as of an instant. Empty if nothing has been written."""
    return query_manifest(dataset_name=dataset_name, root_uri=root_uri, as_of=as_of)


def resolve_manifest(
    *,
    dataset_name: str,
    params: pydantic.BaseModel,
    root_uri: str,
    as_of: datetime.datetime | None = None,
) -> ManifestRow:
    """The newest write of *dataset_name* whose params equal *params* exactly,
    optionally as of an instant.

    Raises:
        KeyError: If no write matches.
    """
    _require_aware("as_of", as_of)

    source = _build_manifest_source(root_uri, dataset_name)
    if source is None:
        LOGGER.info("resolve miss: dataset=%s (no manifest data)", dataset_name)
        msg = f"no {dataset_name!r} found for params: {params.model_dump()}"
        raise KeyError(msg)

    clauses = ["params_json = $1"]
    query_params: list[Any] = [params.model_dump_json()]
    if as_of is not None:
        clauses.append("write_time <= $2")
        query_params.append(as_of)

    rows = _execute_manifest_query(
        root_uri,
        source,
        clauses,
        query_params,
        order_by="ORDER BY write_time DESC",
        limit=1,
    )
    if not rows:
        LOGGER.info(
            "resolve miss: dataset=%s params=%s as_of=%s",
            dataset_name,
            params.model_dump(),
            as_of,
        )
        if as_of is not None:
            msg = f"no {dataset_name!r} found for params as of {as_of.isoformat()}: {params.model_dump()}"
        else:
            msg = f"no {dataset_name!r} found for params: {params.model_dump()}"
        raise KeyError(msg)

    LOGGER.debug(
        "resolved: dataset=%s write_id=%s data_uri=%s",
        dataset_name,
        rows[0].write_id,
        rows[0].data_uri,
    )
    return rows[0]
