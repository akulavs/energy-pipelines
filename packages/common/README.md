# common

Shared building blocks with no dependency on any domain package: the storage
layer, Pydantic/DataFrame base types, and an HTTP client with retries.

## Modules

- [storage/](src/common/storage) — manifest-driven dataset storage and run provenance (below).
- [frames.py](src/common/frames.py) — `BaseDataFrameSchema`, the base for Polars schemas, with a `Cols` namespace so column names are never string literals.
- [models.py](src/common/models.py) — `StrictModel` / `FrozenModel` defaults for `pydantic.BaseModel`, and a case-insensitive `StrEnum`.
- [types.py](src/common/types.py) / [exceptions.py](src/common/exceptions.py) — Pydantic-friendly annotations and the shared exception types.
- [clients/http.py](src/common/clients/http.py) — `httpx.AsyncClient` wrapped in `tenacity` retries.
- [utilities/](src/common/utilities) — JSON and collection helpers.

## Storage

There is no database. Every write drops a Parquet file plus a small JSON
*manifest* sidecar, and reads discover data by scanning the sidecars with DuckDB.
Any `pyarrow.fs` URI works as a root — a local path or `s3://…` — so the same code
runs locally and in the cloud.

A dataset is a `dataset_name` string plus a frozen params model; the params are
the lookup key.

```
{root_uri}/{dataset_name}/<data_id>.parquet             <- data
{root_uri}/_manifests/{dataset_name}/{write_id}.json    <- one manifest per write
{root_uri}/_flows/{flow_id}.json                        <- one record per flow run
```

- [columnar.py](src/common/storage/columnar.py) — `write_dataset` validates a DataFrame against its schema, writes it, and records the manifest; `read_dataset` resolves the manifest and reads back.
- [manifest.py](src/common/storage/manifest.py) — the sidecar itself. `resolve_manifest` returns the newest write matching the params, with optional point-in-time (`as_of`) reads; `scan_manifest` returns the newest write per distinct params.
- [flow_manifest.py](src/common/storage/flow_manifest.py) — one record per flow run: status, `input_ids`, metadata, error. Dataset manifests carry the `flow_id` that produced them, so the two together answer "what produced X?" and "what did run X produce?".
- [catalog.py](src/common/storage/catalog.py) — the registry of what each dataset name *is* (schema + params model). Domain packages register their own datasets; `common` never imports them.
