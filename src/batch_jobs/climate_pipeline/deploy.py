"""
Deploy the climate pipeline flows to a local Prefect server via ``serve()``.

With the Prefect server running (``uv run prefect server start``):

    uv run python src/batch_jobs/climate_pipeline/deploy.py

Registers four deployments and stays running to execute flow runs from the UI or
CLI (Ctrl-C to stop):

- ``ingest-era5/local``           -- ERA5-Land -> bronze (live CDS)
- ``ingest-nsrdb/local``          -- NSRDB -> bronze (live NREL)
- ``era5-nsrdb-silver/local``     -- join bronze -> silver
- ``run-climate-pipeline/local``  -- all three, in order

All data lands under a single ``root_uri`` -- set the ``CLIMATE_PIPELINE_ROOT`` env
var to a folder outside the repo (e.g. a Google Drive Desktop folder) before
serving; nothing machine-specific is committed here.

Dates are ISO strings; Prefect coerces them via the flows' type hints. Points are
[lat, lon] pairs. The bronze ingests need CDS / NSRDB credentials;
``era5_nsrdb_silver`` only reads bronze that already exists.

The ERA5 run form's ``variables`` list and the NSRDB form's ``attributes`` list
are selections: delete the ones a run does not need and it fetches the rest. The
columns left out are stored as nulls, and a later run that puts one back
re-fetches the points that lack it.
"""

from __future__ import annotations

import os

from batch_jobs.climate_pipeline.flows import (
    era5_nsrdb_silver,
    ingest_era5,
    ingest_nsrdb,
    run_climate_pipeline,
)

if __name__ == "__main__":
    import prefect

    # Single root_uri for all climate-pipeline data, kept out of the repo. Set
    # CLIMATE_PIPELINE_ROOT to a path/URI the storage layer accepts (e.g. a Google
    # Drive Desktop folder, or an s3:// prefix); nothing is committed here.
    # Resolved at run time (not import) so importing this module has no side effect.
    root_uri = os.environ.get("CLIMATE_PIPELINE_ROOT")
    if not root_uri:
        raise RuntimeError(
            "set CLIMATE_PIPELINE_ROOT to the external data folder before serving "
            "(e.g. a Google Drive Desktop path); it must live outside the repo"
        )
    # Only ``root_uri`` + ``writer`` are pinned; ``request`` is left unset so the run
    # form is built from the flow schema (``points`` required, dates defaulted).
    params = {"root_uri": root_uri, "writer": "climate_pipeline"}

    deployments = [
        ingest_era5.to_deployment(
            name="local", parameters=params, tags=["climate", "bronze", "era5"]
        ),
        ingest_nsrdb.to_deployment(
            name="local", parameters=params, tags=["climate", "bronze", "nsrdb"]
        ),
        era5_nsrdb_silver.to_deployment(
            name="local", parameters=params, tags=["climate", "silver"]
        ),
        run_climate_pipeline.to_deployment(
            name="local", parameters=params, tags=["climate", "pipeline"]
        ),
    ]
    # Prefect defaults a deployment's description to the flow docstring (an empty
    # string is falsy and still falls back), so blank it on the objects directly
    # -- the UI then shows nothing for description.
    for d in deployments:
        d.description = None  # ty:ignore[invalid-assignment]
    prefect.serve(*deployments)  # ty:ignore[invalid-argument-type]
