# AGENTS.md

## Commands

```bash
script/bootstrap      # install locked dependencies into .venv
script/test           # run tests
script/type-check     # ty
script/lint           # ruff check
script/format         # ruff format
```

Use `uv run` for every Python command -- never bare `python` or `pytest`. If
dependencies change, run `uv lock` and then `script/bootstrap`. Run all four
checks after changes.

## Layout

```
packages/common/         # storage layer, base models -- no domain knowledge
packages/external-data/  # the pipelines: fetch, transform, and store public datasets
src/batch_jobs/          # thin Prefect flows that call into the packages
```

Packages own the logic and never import Prefect or `batch_jobs`. Flows only
resolve inputs, call package functions, and write results through the manifest
system.

## Conventions

- Python 3.14+: `list[int]`, `T | None`.
- Absolute imports.
- `pytest_mock` for mocking, never `unittest.mock`. No `__init__.py` in test directories.
- One `BaseDataFrameSchema` per table shape, with `dtype` and `description` on every field; reference columns as `Schema.Cols.name`, never as string literals.
- One `FrozenModel` params class per dataset, `extra="forbid"`, holding only the fields that distinguish versions.
- Dataset names are module-level constants; never inline one.
