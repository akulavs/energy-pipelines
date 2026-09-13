# dsgrid industrial

The data source behind `external_data.load_pipeline.dsgrid`. This doc describes the
dataset — what it is, where it comes from, what the values mean, and how to use
them. For *how we ingest it*, read the code in this directory (the shared HDF5
fetch / reconstruction machinery lives in [`../dsg_common.py`](../dsg_common.py));
for how the pipeline around it works, the [pipeline README](../README.md).

## What it is

**dsgrid** is the demand-side dataset from the **Electrification Futures Study
(EFS)**, produced by **NLR** (National Laboratory of the Rockies) for the U.S.
Department of Energy. It models hourly U.S. electricity demand at county
resolution, broken out by sector and end use.

This module ingests **both** industrial `.dsg` files, selected by
`DsgridRequestArgs.source`:

| `source` | File | Covers |
|---|---|---|
| `DsgridSource.INDUSTRIAL` (default) | `industrial.dsg` | **manufacturing** demand by 4-digit NAICS **subsector**, split into 12 electricity end uses |
| `DsgridSource.GAPS` | `industrial_gaps.dsg` | the **non-manufacturing** industrial **sectors** (agriculture `11`, mining `21`, construction `23`), a single un-decomposed end use |

Together they complete the industrial total. Both files share the same layout and
the same reconstruction pipeline; only the sector granularity and end-use
breakout differ.

- Overview: <https://data.openei.org/submissions/4130>
- Toolkit / docs: <https://www.nlr.gov/analysis/dsgrid>
- Methodology: EFS report NLR/TP-6A20-71492.

## Where we pull it from

The public **OEDI data lake** S3 bucket over plain HTTPS — no account, API key,
or login, and no rate limits.

- Bucket: `oedi-data-lake`, prefix `dsgrid-2018-efs/raw_complete/`
- Files we ingest: **`industrial.dsg`** (~15 MB) and **`industrial_gaps.dsg`** (~2.6 MB)

Each source produces two bronze datasets, both **parquet**:

| Bronze dataset | Grain |
|---|---|
| `dsgrid_industrial_metadata_bronze` | one row per county × NAICS subsector (annual end-use totals) |
| `dsgrid_industrial_timeseries_bronze` | one row per county × NAICS subsector × hour (hourly load profile) |
| `dsgrid_industrial_gaps_metadata_bronze` | one row per county × NAICS sector (annual electricity total) |
| `dsgrid_industrial_gaps_timeseries_bronze` | one row per county × NAICS sector × hour (hourly load profile) |

One request drives the pipeline: `DsgridRequestArgs(source=…, state=…)`. The
metadata and timeseries datasets for a source always cover the same geography.
The default is `source=INDUSTRIAL`, `state=DC`. Neither table is partitioned: a
write is one state, so a `state=` directory held a single file and repeated its own
key, and the county split that replaced it scattered a state across 56–62 files
(~75 KB each for the gaps source) for a pruning nothing performs. The manifest is
the index, as for every other dataset here.

County remains the geography in the *data* — as fine as this can honestly go.
dsgrid has no PUMA column, and PUMAs do not nest inside counties (DC's one county
holds five). Splitting a county's load across them would need an allocation basis
dsgrid does not publish, and copying it into each would make the industrial total
unrecoverable by summing.

### How the source file is laid out

The `.dsg` is a dsgrid-legacy HDF5 file (`dsgrid=0.2.0`) with two groups:

- `enumerations/` — the labels: `geography` (3234 counties, 5-digit FIPS +
  "County, ST"), `enduse` (12 industrial end uses), `sector` (86 NAICS
  subsectors), `time` (8784 hourly timestamps).
- `data/<naics>/` — per subsector, a compressed representation. The **shapes**
  array `(n, 12, 8784)` holds the `n` *distinct* normalized hourly load profiles
  for that subsector (one curve per end use × 8784 hours). These are the "load
  shapes": the model finds that across the whole country a subsector's hourly
  usage pattern collapses to only a handful of representative curves, so instead
  of storing one profile per county it stores each unique curve once. `n` is
  therefore tiny relative to the county count — e.g. subsector `3231` (Printing)
  has just **5 shapes reused across 2,393 counties**. Each county then carries a
  `geographies (idx → shape, scale)` map: it points at one shape and a scalar
  magnitude. A county's profile is reconstructed as `shape[idx] × scale` (times
  the per-end-use and per-hour scale factors), which is what recovers real MWh
  from the normalized curve. A null index (`4294967295`) means the subsector is
  absent in that county (skipped).

We derive `county_gisjoin` from the FIPS and the two-letter `state` from the
county name's `", ST"` suffix — validated rather than assumed, since that suffix is
the pipeline's whole state selector.

## The values

The curated columns follow the shared building-stock/EFS bronze schema, mapped to
dsgrid's fields.

**`dsgrid_industrial_metadata_bronze`** — key `(county_gisjoin, naics_subsector)`:
`state`, `county_gisjoin`, `county_fips`, `county_name`, `naics_subsector`,
`subsector_name`, `annual_electricity_total_mwh`, and one
`annual_electricity_<end_use>_mwh` per end use.

**`dsgrid_industrial_timeseries_bronze`** — keys `(timestamp, county_gisjoin,
naics_subsector)`: `timestamp` (hourly, interval-ending, naive local standard
time), `county_gisjoin`, `state`, `naics_subsector`, `electricity_total_mwh`, and
one `electricity_<end_use>_mwh` per end use (Float32).

The 12 end uses (all electricity): conventional boiler use, process heating,
process cooling & refrigeration, machine drive, electro-chemical processes, other
process use, facility HVAC, facility lighting, other facility support, on-site
transportation, other non-process use, and end-use-not-reported.
`electricity_total_mwh` is their sum.

### The gaps source (`source=GAPS`)

`industrial_gaps.dsg` covers the **non-manufacturing** industrial sectors that
complete the industrial total (national gaps total ≈ 184 TWh: ~36 agriculture +
~85 mining + ~63 construction). It differs from the manufacturing file in two
ways only:

- **2-digit `naics_sector`** (11 agriculture, 21 mining, 23 construction) instead
  of the 4-digit `naics_subsector`, with a `sector_name` label.
- **A single, un-decomposed end use** — there are no `electricity_<end_use>_mwh`
  breakout columns; `electricity_total_mwh` *is* the sector value.

Everything else (geography columns, 2012 hourly axis, MWh units, partitioning)
matches the manufacturing datasets above.

### Sample of total electricity consumption (DC, `state=DC`)

DC has **26** of the 86 subsectors, totalling **≈162,408 MWh (162.4 GWh) / year**.
Top subsectors by annual total, from `dsgrid_industrial_metadata_bronze`:

| naics_subsector | subsector_name | annual_electricity_total_mwh |
|---|---|--:|
| 3231 | Printing and Related Support Activities | 68,215.0 |
| 3241 | Petroleum and Coal Products Manufacturing | 20,406.9 |
| 3365 | Railroad Rolling Stock Manufacturing | 12,728.5 |
| 3254 | Pharmaceutical and Medicine Manufacturing | 8,947.4 |
| 3219 | Other Wood Product Manufacturing | 7,835.1 |

The hourly `dsgrid_industrial_timeseries_bronze` breaks each of those annual
totals into 8784 interval-ending hours, e.g. subsector 3231 at the start of the
year:

| timestamp | naics_subsector | electricity_total_mwh |
|---|---|--:|
| 2012-01-01 01:00:00 | 3231 | 4.722 |
| 2012-01-01 02:00:00 | 3231 | 4.741 |
| 2012-01-01 03:00:00 | 3231 | 4.759 |

## Caveats worth knowing (verified against the DC baseline)

- **Modeled, not metered.** EFS is a bottom-up demand model. Good for sector
  shape and county allocation, not for validating a specific plant's bill.
- **The grain is county × subsector, not a building.** There is no `bldg_id` and
  no per-building `weight` — the values are already scaled to real-world totals
  (the national reconstructed total ≈ 893 TWh, matching U.S. industrial
  electricity).
- **The weather/time year is 2012**, not 2018 — a leap year, so **8784** hourly
  interval-ending timestamps at a fixed `-05:00` (EST, no DST), stored naive.
- **A subsector is present only in some counties.** Counties where a subsector is
  absent are skipped (not written as zero rows). DC, for example, has 26 of the
  86 subsectors.
- **Electricity only.** This file carries no fuel/gas end uses.
- **The metadata annual total is authoritative.** The `annual_electricity_*_mwh`
  columns are `Float64`; the timeseries `electricity_*_mwh` columns are `Float32`
  for storage. Re-summing the hourly values drifts slightly from the annual
  totals (float rounding) — use the metadata figures when you need the exact annual.
- **`naics_subsector`** is the 4-digit NAICS manufacturing subsector (e.g. `3111`
  = Animal Food Manufacturing); it is the industrial analog of a "building type".
