"""
Parquet datasets with manifest-based discovery.

A dataset is a ``dataset_name`` plus a frozen params model; the params are the
lookup key. Each write lands one Parquet file and one manifest sidecar::

    {root_uri}/{dataset_name}/<data_id>.parquet             <- data
    {root_uri}/_manifests/{dataset_name}/{write_id}.json    <- manifest

Writing::

    write_dataset(df, MySchema, "my_dataset", params, root_uri, writer)

Reading::

    df = read_dataset(MySchema, "my_dataset", params, root_uri)

    # or in two steps, to inspect the manifest row first
    row = resolve_manifest(dataset_name="my_dataset", params=params, root_uri=root_uri)
    df = read_parquet(row.data_uri, MySchema)
"""

from __future__ import annotations

import datetime
import logging
import uuid

import polars as pl
import pydantic

from common.frames import BaseDataFrameSchema
from common.storage.manifest import (
    ManifestRow,
    _ensure_local_parent_dir,
    _warn_if_mutable_params,
    resolve_manifest,
    write_manifest,
)

LOGGER = logging.getLogger(__name__)


def write_dataset(
    df: pl.DataFrame,
    schema: type[BaseDataFrameSchema],
    dataset_name: str,
    params: pydantic.BaseModel,
    root_uri: str,
    writer: str,
    write_time: datetime.datetime | None = None,
    flow_id: str = "",
) -> ManifestRow:
    """Validate *df* against *schema*, write it as Parquet, and record the manifest.

    Returns the :class:`ManifestRow`, so the caller has the generated ``write_id``.
    """
    _warn_if_mutable_params(params)
    if write_time is not None and write_time.tzinfo is None:
        msg = "write_time must be timezone-aware (e.g. datetime.datetime.now(tz=datetime.UTC))"
        raise ValueError(msg)

    schema.validate(df)

    relative_uri = f"{dataset_name}/{uuid.uuid4()}.parquet"
    absolute_uri = f"{root_uri}/{relative_uri}"
    _ensure_local_parent_dir(absolute_uri)
    df.write_parquet(absolute_uri)

    return write_manifest(
        write_time=write_time,
        data_uri=relative_uri,
        params=params,
        root_uri=root_uri,
        dataset_name=dataset_name,
        writer=writer,
        flow_id=flow_id,
    )


def read_parquet(data_uri: str, schema: type[BaseDataFrameSchema]) -> pl.DataFrame:
    """Read the Parquet file at *data_uri* and validate it against *schema*.

    Raises:
        RuntimeError: If the file is missing -- the manifest says a write
            completed, so an absent file means it failed mid-run.
    """
    try:
        df = pl.read_parquet(data_uri)
    except FileNotFoundError:
        msg = f"data is missing at {data_uri!r}; the write may have failed mid-run"
        raise RuntimeError(msg) from None
    schema.validate(df)
    return df


def read_dataset(
    schema: type[BaseDataFrameSchema],
    dataset_name: str,
    params: pydantic.BaseModel,
    root_uri: str,
    *,
    as_of: datetime.datetime | None = None,
) -> pl.DataFrame:
    """Resolve the manifest for *params*, then read and validate in one call."""
    row = resolve_manifest(
        dataset_name=dataset_name,
        params=params,
        root_uri=root_uri,
        as_of=as_of,
    )
    return read_parquet(row.data_uri, schema)
