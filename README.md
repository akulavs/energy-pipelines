# energy-pipelines

Data pipelines that fetch public climate and building-load datasets, land them as
versioned Parquet, and join them into analysis-ready tables.

- **Climate** — ERA5-Land hourly weather (Copernicus CDS) and NSRDB solar
  irradiance (NREL), joined onto ERA5's 0.1° grid. See
  [`climate_pipeline/README.md`](packages/external-data/src/external_data/climate_pipeline/README.md).
- **Load** — ResStock and ComStock building-stock energy (OEDI) per PUMA, plus
  dsgrid industrial demand per state, stacked into an industrial silver table. See
  [`load_pipeline/README.md`](packages/external-data/src/external_data/load_pipeline/README.md).
- **Data fetch** — helpers that return the climate datasets for a set of points, or
  the load datasets for a set of PUMAs, running the pipelines only for what the
  store does not already hold.

Every source is public. The CDS and NREL bronze ingests need free API credentials;
OEDI needs none.

## Getting started

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. `script/bootstrap` — installs Python 3.14 and the locked dependencies into `.venv`.
3. `script/test` — the suite runs offline against small fixtures; tests that reach a
   live service are marked `integration` and skipped by default.

Other scripts: `script/lint`, `script/format`, `script/type-check`. All of them run
through `uv run --locked`; if you change dependencies, run `uv lock` first.

## Layout

```
packages/
  common/           # storage layer: manifest-backed Parquet datasets, run provenance
  external-data/    # the pipelines themselves -- schemas, fetchers, transforms
src/batch_jobs/     # thin Prefect flows that orchestrate the packages
```

Packages hold all the logic and have no Prefect dependency, so they work from a
plain script or notebook. The flows only resolve inputs, call package functions,
and record what they wrote. To serve them locally:

```bash
uv run prefect server start
CLIMATE_PIPELINE_ROOT=~/code/energy-pipelines-data/climate_pipeline uv run python src/batch_jobs/climate_pipeline/deploy.py
LOAD_PIPELINE_ROOT=~/code/energy-pipelines-data/load_pipeline       uv run python src/batch_jobs/load_pipeline/deploy.py
```

Each root is any path or `s3://` prefix the storage layer accepts. The defaults
above keep the data next to the repo in `~/code/energy-pipelines-data`, one
subfolder per pipeline, outside the repo so nothing machine-specific is committed.
The storage layer creates the folders on first write. Open
http://127.0.0.1:4200/deployments to trigger runs.

## Storage

There is no database. Each dataset write lands one Parquet file and one JSON
manifest sidecar under a root URI (a local folder or `s3://` prefix); reads scan
the sidecars with DuckDB and resolve the newest write matching a request, with
optional point-in-time (`as_of`) reads. Each flow run also records what it read
and wrote, so lineage runs in both directions. The details are in
[`packages/common/README.md`](packages/common/README.md).
