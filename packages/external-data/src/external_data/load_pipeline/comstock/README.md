# ComStock

The data source behind `external_data.load_pipeline.comstock`. This doc describes
the dataset — what it is, where it comes from, what the values mean, and how to use
them. For *how we ingest it*, read the code in this directory (the OEDI fetch /
request base / timeseries schema / PUMA assembly shared with ResStock live in
[`../oedi_building_stock.py`](../oedi_building_stock.py)); for how the pipeline
around it works, the [pipeline README](../README.md).

## What it is

**ComStock** is a synthetic model of the U.S. **commercial** building stock
produced by **NLR** (National Laboratory of the Rockies) for the **U.S.
Department of Energy**, part of the *End-Use Load Profiles for the U.S. Building
Stock* project. Roughly 350k physics-based building-energy simulations
(OpenStudio / EnergyPlus), calibrated against measured AMI data, model what is
actually drawing power behind a service territory — and let us isolate the
weather-sensitive (cooling) share of load and forecast electrification-driven
growth.

- Overview: <https://comstock.nlr.gov/page/datasets>
- Underlying files: [OEDI Data Lake](https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=nrel-pds-building-stock%2Fend-use-load-profiles-for-us-building-stock%2F)

## Where we pull it from

The public **OEDI data lake** S3 bucket over plain HTTPS — no account, API key,
or login, and no rate limits.

- Bucket: `oedi-data-lake`, prefix
  `nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/`
- Release we ingest: **`2025/comstock_amy2018_release_3`** (the latest, Nov 2025)

Two bronze datasets, **parquet**, mirroring ResStock's pair so the two sources read
alike — one row per building, one row per building × timestep, joined on `bldg_id`:

| Bronze dataset | OEDI source | Grain |
|---|---|---|
| `comstock_puma_metadata_bronze` | `metadata_and_annual_results_aggregates/by_state_and_puma/full/parquet/state=…/puma=…/{STATE}_{PUMA}_upgrade{N}_agg.parquet` (one file per PUMA) | one row per building, census-tract duplication collapsed |
| `comstock_timeseries_bronze` | many `timeseries_individual_buildings/by_state/upgrade={N}/state=…/{bldg_id}-{N}.parquet` files (one per building) | one row per building × 15-min timestep, for every building in a PUMA |

Despite the family name, `metadata_and_annual_results_aggregates` is **not** a
rollup: "aggregates" describes how the *rows* were combined, not the values. A
building appears once, carrying the `weight` that the per-county file splits across
every census tract it represents. Measured on DC PUMA `G11000101` — 946 rows here
against 2,985 tract-duplicated rows there for the same 946 buildings, with
per-building `weight` equal to the sum of the split weights.

`weight` is scoped to **that PUMA**: a building represents stock across many PUMAs
(one California model spans 50), and the file carries only this PUMA's share.
Correct for PUMA-grain work; wrong to sum across PUMAs.

The per-building table is assembled **client-side, one PUMA at a time**: read the
PUMA metadata for its `bldg_id`s, then fetch each building's timeseries file
concurrently and concat. Neither dataset is partitioned into directories — every
write already names one PUMA, and the manifest is the index. `bldg_id` is a
join-key column, not a request field.

The default geography is Washington, DC (`state=DC`, PUMA `G11000101`), baseline
scenario (`upgrade=0`). The state is derived from the PUMA's FIPS prefix.

**What is deliberately not ingested.** The release holds ten file families and we
take **two**. Also published: `metadata_and_annual_results` (per county,
tract-duplicated), `timeseries_aggregates` (pre-weighted, no `bldg_id`), `weather`
(the AMY2018 series the simulations consumed, per county), `geographic_information`
(PUMA boundary geometry — 52 MB of MultiPolygons, which is what a lat/long front
door would need), `commercial_gap_model` (stock the main model does not cover —
worth checking before treating this source as complete), plus `component_loads`,
`building_energy_models` and `comparison_plots`.

The aggregate path is the interesting one to have refused: it is ~15 requests for a
PUMA against ~950 and is the cheaper answer to "what does this location draw" — but
it carries no `bldg_id`, so it cannot answer "which buildings drew it", and an
aggregate cannot be un-summed. Choosing the per-building pair buys segmentation by
vintage, HVAC type or fuel at ~21× the bytes.

> `timeseries_aggregates` is published `by_puma`, `by_county`, `by_state`,
> `by_iso_rto_region` and two climate-zone axes. **ResStock publishes no PUMA or
> county axis** — only state and the region/zone ones — which is why residential
> load at PUMA grain has no aggregate path and must be assembled per building.

## The values

Both bronze tables rename the raw dotted, unit-suffixed OEDI columns to clean
identifiers (`in.sqft..ft2` → `sqft`). All energy is delivered in **kWh — no
conversion**. The per-PUMA metadata file carries ~1,300 columns; the curated set is
the 14 below.

**`comstock_puma_metadata_bronze`** — key `bldg_id`:

| Column | Raw OEDI column | What it is |
|---|---|---|
| `bldg_id` | `bldg_id` | building model id; the join key to the timeseries, unique in this table |
| `upgrade` | `upgrade` | measure-package id (0 = baseline) |
| `state` | `in.state` | two-letter state code |
| `puma_gisjoin` | `in.nhgis_puma_gisjoin` | the PUMA this file covers |
| `building_type` | `in.comstock_building_type` | ComStock commercial type (e.g. `Warehouse`, `SmallOffice`) |
| `vintage` | `in.vintage` | building age cohort |
| `hvac_system_type` | `in.hvac_system_type` | commercial HVAC system type |
| `heating_fuel` | `in.heating_fuel` | primary heating fuel |
| `sqft` | `in.sqft..ft2` | modelled floor area (ft²) |
| `weighted_sqft` | `calc.weighted.sqft..ft2` | floor area at real-world scale (= `sqft × weight`) |
| `weight` | `weight` | buildings represented, **within this PUMA** |
| `annual_electricity_total_kwh` | `out.electricity.total.energy_consumption..kwh` | annual total electricity, unweighted |
| `annual_electricity_cooling_kwh` | `out.electricity.cooling.energy_consumption..kwh` | annual cooling electricity |
| `annual_electricity_heating_kwh` | `out.electricity.heating.energy_consumption..kwh` | annual electric-heating electricity |

**No county, tract or climate-zone columns, deliberately** — though the file has
them. Every geography column in this file is prefixed `in.as_simulated_`: the
county, tract and climate zone the model was *simulated* in, which is not where its
load sits. For DC PUMA `G11000101` the simulated counties include two in Virginia,
so a column named `county_gisjoin` would hold another state's county. The location
stays the PUMA the file is keyed by. A reader needing the counties a PUMA overlaps
gets them from `resstock_metadata_bronze`, which carries both codes.

**`comstock_timeseries_bronze`** — keys `timestamp`, `bldg_id`: `timestamp`
(15-min, interval-ending, local standard time), `bldg_id`, `state`, `puma_gisjoin`
(the PUMA the buildings were assembled for), and the three `electricity_*_kwh`
columns **per interval** (Float32). One dataset holds every building in the PUMA,
keyed by that PUMA.

Annual metadata totals are Float64 while the per-interval columns are Float32:
35,040 intervals a year get summed, so the annual figures are the exact ones.

## Caveats worth knowing (verified against the DC baseline)

- **Synthetic, not metered.** Good for shape and segmentation, not for
  validating a real site's bill.
- **One row per building here — the tract duplication is already collapsed.** The
  release's *per-county* metadata file repeats a `bldg_id` once per census tract
  (DC: 19,216 rows for 1,662 buildings, up to 157 rows each), so summing energy
  across its raw rows over-counts ~6×. This file is the reason not to touch that
  one: joining a tract-duplicated table to a timeseries fans each profile out by
  its tract count — 6.2× on average within a PUMA, and 1,147 rows for one
  California building.
- **`weight` is an expansion weight — multiply by it and sum.** It's the number of
  real buildings a model stands in for *within this PUMA*, so a real-world total is
  `Σ (weight × value)`, not a raw row count. Measured on CA PUMA `G06009703`, 844 of
  864 buildings represent more stock outside the PUMA than in — so this is not the
  model's global weight, but it is exactly the share this PUMA's load should
  reflect. For floor area use `weighted_sqft`, not `sqft × weight`, and don't
  double-apply.
- **Zeros are real, not missing.** Cooling/heating electricity is `0` for
  buildings on gas or other fuels.
- **Enums are CamelCase** (`RetailStripmall`, `NaturalGas`).
- **Building ids reset every release** — compare by segment/characteristics, not
  by `bldg_id`, across releases.
- **Timeseries is 15-minute, not hourly** — 35,040 intervals per building per
  year (365 × 24 × 4), interval-ending, wide format. Resample downstream for 8,760.
- **Full per-building timeseries is huge** (hundreds of GB per release); we pull
  a PUMA's worth of building files at a time (one HTTP fetch per building,
  concurrent), not the whole set.
