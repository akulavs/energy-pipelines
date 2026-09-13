# external-data

Clients and models for external data sources.

This distribution ships a single top-level importable module, `external_data`
(see `[tool.uv.build-backend] module-name` in `pyproject.toml`), containing:

  - **`external_data.climate_pipeline`** — the unified ERA5-Land × NSRDB climate
    pipeline. Ingests ERA5-Land weather (`era5_weather_bronze`, from CDS) and NSRDB
    solar (`nsrdb_solar_bronze`, from NREL), then joins them into
    `era5_nsrdb_silver`: one
    harmonized hourly row per ERA5-Land 0.1° grid node — ERA5 weather in friendly
    units (°C, wind speed/direction, mm precip) plus NSRDB solar/irradiance snapped
    onto the grid (`None` where NSRDB doesn't cover the node/hour). NSRDB bronze is
    keyed by fetch interval, and the join reads the 60-minute slice to match ERA5's
    hourly grid. See the source docs:
    [`climate_pipeline/era5/README.md`](src/external_data/climate_pipeline/era5/README.md)
    and
    [`climate_pipeline/nsrdb/README.md`](src/external_data/climate_pipeline/nsrdb/README.md).

  - **`external_data.load_pipeline`** — the three sector load sources, landed as
    bronze per PUMA and stacked into industrial silver. See the
    [pipeline README](src/external_data/load_pipeline/README.md) for the budgets,
    the coverage rules, and how failures are handled.

      - **`load_pipeline.resstock`** — **ResStock** (synthetic residential
        building-stock energy) from the public OEDI data lake. Two datasets:
        `resstock_metadata_bronze` (per-building characteristics + annual end-use
        totals) and `resstock_timeseries_bronze` (15-min load profiles for every
        building in a PUMA, assembled from the individual-building files). See
        [`load_pipeline/resstock/README.md`](src/external_data/load_pipeline/resstock/README.md).

      - **`load_pipeline.comstock`** — **ComStock** (synthetic commercial
        building-stock energy), same two datasets in the same shape:
        `comstock_metadata_bronze` and `comstock_timeseries_bronze`. See
        [`load_pipeline/comstock/README.md`](src/external_data/load_pipeline/comstock/README.md).

      - **`load_pipeline.dsgrid`** — **dsgrid** industrial demand from the
        Electrification Futures Study, published per state as two `.dsg` HDF5
        files that the pipeline stacks into one metadata and one timeseries
        table. See
        [`load_pipeline/dsgrid/README.md`](src/external_data/load_pipeline/dsgrid/README.md).

  - **`external_data.climate_pipeline.weather`** — demoware Open-Meteo client
    (reference only, not production).

`external_data.load_pipeline.oedi_building_stock` is not a data source: it holds the
shared OEDI building-stock fetch / request / timeseries-assembly machinery used by
the `resstock` and `comstock` modules above.

Like the rest of the repo, these modules carry no Prefect dependency.
