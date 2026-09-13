# ResStock

The data source behind `external_data.load_pipeline.resstock`. This doc describes
the dataset — what it is, where it comes from, what the values mean, and how to use
them. For *how we ingest it*, read the code in this directory (the OEDI fetch /
request base / timeseries schema / PUMA assembly shared with ComStock live in
[`../oedi_building_stock.py`](../oedi_building_stock.py)); for how the pipeline
around it works, the [pipeline README](../README.md).

## What it is

**ResStock** is a synthetic model of the U.S. **residential** building stock
produced by **NLR** (National Laboratory of the Rockies) for the **U.S.
Department of Energy**, part of the *End-Use Load Profiles for the U.S. Building
Stock* project (~550k physics-based OpenStudio / EnergyPlus simulations
calibrated against measured AMI data).

- Overview: <https://resstock.nlr.gov/datasets>
- Underlying files: [OEDI Data Lake](https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=nrel-pds-building-stock%2Fend-use-load-profiles-for-us-building-stock%2F)

## Where we pull it from

The public **OEDI data lake** S3 bucket over plain HTTPS — no account, API key,
or login, and no rate limits.

- Bucket: `oedi-data-lake`, prefix
  `nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/`
- Release we ingest: **`2025/resstock_amy2018_release_1`** (the latest)

Two bronze datasets, both **parquet**:

| Bronze dataset | OEDI source | Grain |
|---|---|---|
| `resstock_metadata_bronze` | `metadata_and_annual_results/by_state/full/parquet/state=…/{STATE}_upgrade{N}.parquet` (one file per state) | one row per building **of one PUMA** |
| `resstock_timeseries_bronze` | many `timeseries_individual_buildings/by_state/upgrade={N}/state=…/{bldg_id}-{N}.parquet` files (one per building) | one row per building × 15-min timestep, for every building in a PUMA |

The metadata is one file per state, but the **write is per PUMA**: the state file
is read once per run and a slice cut from it for each PUMA asked for, so the key
names the PUMA whose buildings it holds. The trade is that a PUMA ingested in a
later run re-downloads the 54 MB state file to cut its own slice — the price of
every dataset being addressable by a PUMA.

The individual-building timeseries is one file **per building**, so
`resstock_timeseries_bronze` is assembled **client-side, one PUMA at a time**: read
that PUMA's metadata for its `bldg_id`s, then fetch each building's timeseries file
concurrently and concat. Neither dataset is partitioned into directories; the
manifest is the index. `bldg_id` is a join-key column, not a request field.

The default geography is Washington, DC (`state=DC`), PUMA `G11000101`, baseline
scenario (`upgrade=0`).

ResStock's metadata is also the pipeline's **PUMA → county** lookup: it is the only
bronze table carrying both `puma_gisjoin` and `county_gisjoin`, which is what a
reader needs to line a PUMA up against the county-grained dsgrid tables.

## The values

Bronze renames the raw dotted, unit-suffixed OEDI columns to clean identifiers
(`in.sqft..ft2` → `sqft`). All energy is delivered in **kWh — no conversion**.
The curated columns follow the shared data-source schema, mapped to ResStock's
field names.

**`resstock_metadata_bronze`** — key `bldg_id`:

| Column | Raw OEDI column |
|---|---|
| `bldg_id` | `bldg_id` |
| `county_gisjoin`, `county_name`, `puma_gisjoin` | `in.county`, `in.county_name`, `in.puma` |
| `building_type` | `in.geometry_building_type_recs` |
| `vintage` | `in.vintage` |
| `hvac_cooling_type` | `in.hvac_cooling_type` |
| `heating_fuel` | `in.heating_fuel` |
| `climate_zone` | `in.ashrae_iecc_climate_zone_2004` |
| `sqft` | `in.sqft..ft2` |
| `weight` | `weight` |
| `annual_electricity_total_kwh`, `annual_electricity_cooling_kwh`, `annual_electricity_heating_kwh` | `out.electricity.{total,cooling,heating}.energy_consumption..kwh` |

**`resstock_timeseries_bronze`** — keys `timestamp`, `bldg_id`: `timestamp`
(15-min, interval-ending, local standard time), `bldg_id`, `state`, `puma_gisjoin`
(the PUMA the buildings were assembled for), and the same three
`electricity_*_kwh` columns **per interval** (Float32). One dataset holds every
building in the PUMA, keyed by that PUMA.

Annual metadata totals are Float64 while the per-interval columns are Float32:
35,040 intervals a year get summed, so the annual figures are the exact ones.

## Caveats worth knowing (verified against the DC baseline)

- **Synthetic, not metered.** Good for shape and segmentation, not for
  validating a real dwelling's bill.
- **One row per building** — Row key is `bldg_id`.
- **`weight` is a state-level expansion weight.** It's the number of real
  dwellings each modeled building stands in for: multiply by it and sum to scale
  the sample up to a real-world **state** total (`Σ weight × value`) — it is
  **not** a percentage or share. It's a single **uniform** value per
  state, so it rescales totals but never the relative mix. For sampling-based
  aggregation you don't apply it directly — sampling with replacement reproduces the
  weighting naturally, since drawing a profile *k* times contributes *k*× its load.
- **Zeros are real, not missing.** Cooling/heating electricity is `0` for
  dwellings on other fuels.
- **Timeseries is 15-minute, not hourly** — **35,040** intervals per building
  per year (365 × 24 × 4), interval-ending, in wide format (one row per interval,
  each end use its own column). Resample to hourly downstream if you need 8,760.
- **Enums are spaced Title Case** ("Multi-Family with 5+ Units", "Room AC",
  "Natural Gas").
- **`heating_fuel` / `hvac_cooling_type` use a literal `"None"` string** for the
  no-heating / no-cooling case (not a null).
- **Climate zone:** use `in.ashrae_iecc_climate_zone_2004` (we do);
  `in.cec_climate_zone` is fully null outside CA.
- **No `calc.weighted.sqft`** in this release — the shared-schema field is absent
  here, so it is not ingested. ComStock has it as `weighted_sqft`.
- **Building ids reset every release** — compare by segment/characteristics, not
  by `bldg_id`, across releases.
- ResStock has a dedicated **EV-charging** end use in the timeseries file
  (`out.electricity.ev_charging…`). Not in the curated bronze
  set today, but available for electrification modeling.
