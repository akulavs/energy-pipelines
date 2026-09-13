"""
Deploy the load pipeline flows to a local Prefect server via ``serve()``.

Set ``LOAD_PIPELINE_ROOT`` to a data folder outside the repo, start the server
(``uv run prefect server start``), then::

    uv run python src/batch_jobs/load_pipeline/deploy.py

Two concurrency knobs are read from the environment, because their right values
depend on the host rather than on the pipeline (``flows.py`` carries the
arithmetic):

- ``LOAD_PIPELINE_OEDI_IN_FLIGHT`` (default 28) -- building fetches in flight. 28
  is this machine's *bandwidth* ceiling: a ~6.3 MB file every ~10 s is already
  ~126 Mbit/s. A host with a fatter pipe should raise it **and** the matching
  ``oedi-api`` global limit on the server, since the smaller of the two wins.
- ``LOAD_PIPELINE_PUMAS_IN_FLIGHT`` (default 1) -- PUMAs fetched at once. One,
  because a PUMA is a single write and its table is resident until it is written
  (~1.7 GB for both sources, roughly double at the concat). Size it from the host's
  memory, not its cores: ``flows.py`` carries the measured table, and 2 fits a 24 GB
  machine with headroom. This is the knob that decides whether several PUMAs are
  fetched together or one after another -- the metadata phase already fans out on
  its own.

Registers five deployments and stays running to serve flow runs (Ctrl-C to stop):

- ``ingest-resstock/local``        -- residential -> bronze
- ``ingest-comstock/local``        -- commercial -> bronze
- ``ingest-dsgrid/local``          -- industrial -> bronze
- ``industrial-load-silver/local`` -- stack the two dsgrid halves -> silver
- ``run-load-pipeline/local``      -- all four, in order

Every run form asks for ``geographies`` (required) plus that flow's fetch config.
``geographies`` is one or more PUMA GISJOIN codes; each state is derived from its
FIPS prefix, so it is never typed separately. Through a deployment each entry is an
object -- ``{"pumas": [{"puma_gisjoin": "G11000101"}]}`` -- which is also where a
PUMA in a state's minority time zone sets its own offset. They need not share a
state: the state-published steps (dsgrid bronze, industrial silver) collapse the
list to its distinct states.

OEDI is public, so no credentials are needed -- but the building-stock ingests fetch
one file per building and a location means every building in it, so budget a few
thousand requests **per PUMA** times the length of the list. The building count is
logged before each PUMA's fetches start.
"""

from __future__ import annotations

import os

from batch_jobs.load_pipeline.flows import (
    industrial_load_silver,
    ingest_comstock,
    ingest_dsgrid,
    ingest_resstock,
    run_load_pipeline,
)

if __name__ == "__main__":
    import prefect

    # Any path/URI the storage layer accepts (a Google Drive Desktop folder, an
    # s3:// prefix). Read at run time, not import, so importing has no side effect.
    root_uri = os.environ.get("LOAD_PIPELINE_ROOT")
    if not root_uri:
        raise RuntimeError(
            "set LOAD_PIPELINE_ROOT to the external data folder before serving "
            "(e.g. a Google Drive Desktop path); it must live outside the repo"
        )
    # Pin only what is machine-specific; the rest of each run form comes from the
    # flow signature, so geography stays a required field the operator fills in.
    params = {"root_uri": root_uri, "writer": "load_pipeline"}

    deployments = [
        ingest_resstock.to_deployment(
            name="local", parameters=params, tags=["load", "bronze", "resstock"]
        ),
        ingest_comstock.to_deployment(
            name="local", parameters=params, tags=["load", "bronze", "comstock"]
        ),
        ingest_dsgrid.to_deployment(
            name="local", parameters=params, tags=["load", "bronze", "dsgrid"]
        ),
        industrial_load_silver.to_deployment(
            name="local", parameters=params, tags=["load", "silver", "industrial"]
        ),
        run_load_pipeline.to_deployment(
            name="local", parameters=params, tags=["load", "pipeline"]
        ),
    ]
    # Prefect defaults a deployment's description to the flow docstring, and an empty
    # string still falls back, so blank it on the objects directly.
    for d in deployments:
        d.description = None  # ty:ignore[invalid-assignment]
    prefect.serve(*deployments)  # ty:ignore[invalid-argument-type]
