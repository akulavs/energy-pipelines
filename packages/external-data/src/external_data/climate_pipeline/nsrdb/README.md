# NSRDB (GOES Aggregated)

The data source behind `external_data.climate_pipeline.nsrdb`. This doc describes
the dataset — what it is, where it comes from, what the values mean, and how to
use them. For *how we ingest it*, read `bronze.py` in this directory.

## What it is

The **National Solar Radiation Database (NSRDB)** is a satellite-derived record
of solar irradiance and supporting meteorology produced by **NREL** (National
Renewable Energy Laboratory) for the **U.S. Department of Energy**. It is the
standard free solar-resource dataset for U.S. PV/solar modeling.

We ingest the **GOES Aggregated, PSM v4.0.0** product:

- Coverage: **1998–2024**, GOES East + West (Americas)
- Spatial resolution: **4 km**
- Temporal resolution: **hourly (60 min) by default**; 30 or 60 minutes supported
- Overview: <https://www.nlr.gov/hpc/nsrdb-dataset>
- API docs: <https://developer.nlr.gov/docs/solar/nsrdb/nsrdb-GOES-aggregated-v4-0-0-download/>

## Where we pull it from

The NSRDB download API (CSV endpoint), which returns a single point for a single
year:

- Dataset id: `nsrdb-GOES-aggregated-v4-0-0`
- Endpoint: `https://developer.nlr.gov/api/nsrdb/v2/solar/nsrdb-GOES-aggregated-v4-0-0-download.csv`

A **free API key** is required (`NSRDB_API_KEY` + `NSRDB_EMAIL`, or a
`~/.nsrdbrc` file) — get one at <https://developer.nlr.gov/signup/>. The `.csv`
endpoint is restricted to **one point and one year per request**; larger pulls
use the async JSON workflow (not implemented here).

> Note: NREL retired the `developer.nrel.gov` domain on 2026-05-29; the API now
> lives at `developer.nlr.gov` (same paths).

### Fetching multiple points over a date range

`ingest_bronze(request)` takes a `ClimatePipelineRequestArgs` (points + date
range) and expands it to one fetch per **(point × spanned calendar year)** —
e.g. 3 points over 2021-06-01→2023-02-01 is `3 × 3 = 9` requests — reusing one
HTTP client. It then **combines** each point's years, **trims** them back to the
request's `[start_date, end_date]` (the API returns whole years), and writes
**one dataset per point**, each with its own manifest row — so the 3 points above
produce 3 entries, not one. Exact-duplicate points are dropped, and an empty
result after trimming fails loud. The NSRDB fetch config (interval / attributes)
is a separate `NsrdbRequestArgs`, defaulting to the curated set — see
[The values](#the-values) for asking it for a subset. Keeping the
range within NSRDB coverage (1998–2024) is the join layer's job — `silver.py`
clamps it; calling this directly with an out-of-range year lets the NSRDB API
reject it.

The API unit is a point-*year*; the **manifest** unit is a point. Writing per
point is what makes a re-run incremental — `read_points_bronze` looks each point
up in the manifest, so a batch-keyed entry (several points under one key) cannot
be resolved and would be invisible to the silver join.

A year the provider **permanently** cannot serve (a 4xx) ends that point's range
rather than discarding it: the years already fetched are written under a key
narrowed to what they cover, so the entry never claims a span it lacks. A
*transient* year failure fails the whole point instead, so a passing blip is never
recorded as a permanent hole.

> **Errors:** NSRDB reports both real rejections and its own backend failures as
> **400**, so the status alone does not classify. Measured against the live API:
> an ocean point or impossible latitude returns
> `["No data available at the provided location", "Data processing failure."]`; a
> year outside coverage returns `["Invalid value(s)"]`; and a request that
> succeeds unchanged on retry returns `["Data processing failure."]` alone. The
> ingest therefore treats the first two phrases (and 401/403/404) as permanent and
> everything else as retryable — keying on the generic phrase in either direction
> would be wrong.

> **Rate limits:** the request count is `points × years`. NSRDB publishes **1,000
> requests/hour per key** on a rolling window
> ([docs](https://developer.nlr.gov/docs/rate-limits/)), returning **429** once
> exceeded; no per-second limit is documented. A key may be granted a different
> quota — every response carries `X-RateLimit-Limit` / `X-RateLimit-Remaining`,
> which `observed_rate_budget` reads and the fetch logs, so check those rather
> than assume. `download_csv` retries 429 (honoring `Retry-After`) and 5xx with
> exponential backoff. The pipeline caps the rate server-side; see the
> [pipeline README](../README.md#provider-budgets).

## How the source produces it (the Physical Solar Model)

NSRDB irradiance is **not measured on the ground** — it is *modeled from
satellite imagery* via NREL's **Physical Solar Model (PSM)**. The pipeline:

1. **Cloud detection** — the **PATMOS-X** model reads visible + infrared
   radiance channels from the **GOES** geostationary satellites to build a cloud
   mask and retrieve cloud properties.
2. **Clear-sky irradiance** — computed with the **REST2** radiative-transfer
   model.
3. **Cloudy-sky irradiance** — where the cloud mask flags cloud, NREL's **FARMS**
   (Fast All-sky Radiation Model for Solar applications), coupled to REST2,
   computes GHI, and FARMS-DNI computes DNI.
4. **Ancillary inputs** — aerosol optical depth, precipitable water, albedo, plus
   the reported **temperature and wind**, come from NASA's **MERRA-2**
   reanalysis.

So each row is a physically-modeled estimate, not a station reading — validated
against ground stations but carrying model uncertainty (largest under broken
cloud, snow, and at high latitudes / low sun angles).

| | |
|---|---|
| Producer | NREL / U.S. DOE |
| Method | Satellite-derived (PSM: PATMOS-X + REST2 + FARMS), ancillary from MERRA-2 |
| Source imagery | GOES East + West |
| Resolution | 4 km spatial; 30- or 60-minute temporal (we default to 60) |
| Coverage | 1998–2024 |

## The values

One row per timestamp for a single 4 km grid cell (the requested point snaps to
the nearest cell). All values are stored **as delivered — no unit conversion**.
Keys: `valid_time`, `latitude`, `longitude`.

A fetch may ask for **any non-empty subset** of the 14 (`NsrdbRequestArgs.attributes`);
an unrecognised name is rejected. The stored table always has all 14 columns —
the ones not fetched are null, each keeping its declared type — so bronze has one
schema whatever a run selected, and the selection is recorded on the write's
manifest entry alongside the interval rather than inferred from its columns. A
later run asking for more re-fetches the point; asking for fewer reuses what is
there. Unlike ERA5, NSRDB bills the *request* rather than the attribute, so a
subset buys a narrower table rather than a cheaper call. A subset propagates into
silver as null columns (the join reads bronze as it finds it), so drop an
attribute only when nothing downstream needs it.

| Column | Variable | Units | What it is / how it's used |
|---|---|---|---|
| `ghi` | Global horizontal irradiance | W m⁻² | Total sun on a flat surface — primary driver of fixed/rooftop **PV energy yield**. |
| `dni` | Direct normal irradiance | W m⁻² | Beam component — **tracking PV** and concentrating solar (CSP). |
| `dhi` | Diffuse horizontal irradiance | W m⁻² | Scattered/sky component — tilted-plane transposition, cloudy-day performance. |
| `clearsky_ghi` | Clear-sky GHI | W m⁻² | Theoretical cloud-free GHI — clear-sky index, curtailment/forecast benchmark. |
| `clearsky_dni` | Clear-sky DNI | W m⁻² | Cloud-free DNI baseline. |
| `clearsky_dhi` | Clear-sky DHI | W m⁻² | Cloud-free DHI baseline. |
| `cloud_type` | NSRDB cloud-type code | code (int) | Categorical cloud class — drives irradiance variability / ramps. |
| `fill_flag` | Fill-flag code | code (int) | **Data-quality flag** marking gap-filled/interpolated values. |
| `air_temperature` | 2 m air temperature | °C | PV cell-temperature derating (efficiency falls as panels heat); demand. |
| `wind_speed` | Wind speed | m s⁻¹ | Module cooling (lowers cell temp → higher yield); wind-resource proxy. |
| `surface_albedo` | Surface albedo | fraction | Ground reflectance — **bifacial PV** rear-side gain. |
| `solar_zenith_angle` | Solar zenith angle | degrees | Sun position — angle-of-incidence, tracking, plane-of-array transposition. |
| `relative_humidity` | Relative humidity | % | Atmospheric attenuation, soiling context. |
| `surface_pressure` | Surface pressure | mbar (hPa) | Air-mass / attenuation term in irradiance modeling. |

### Dropped in the silver join

The [silver join](../silver.py) keeps NSRDB's solar/irradiance columns (`ghi`,
`dni`, `dhi`, the three `clearsky_*`, `solar_zenith_angle`, `surface_albedo`)
plus `relative_humidity` and `surface_pressure`, and **drops**:

- `air_temperature` and `wind_speed` — superseded by ERA5-Land's `t2m` and
  `u10`/`v10`, which silver uses for temperature and wind.
- `cloud_type` and `fill_flag` — integer QA codes (a mean over the hourly
  resample would be meaningless). Dropping `fill_flag` in particular means
  silver's hourly irradiance can no longer be traced to whether a source record
  was gap-filled (`fill_flag != 0`); check `fill_flag` here in bronze if that
  matters for your use.

### Conventions to know

- **Irradiance is instantaneous** at the timestamp (W m⁻²), **not** accumulated
  over the interval — there is **nothing to de-accumulate**.
- **Temperature is °C**; pressure is mbar (= hPa).
- `cloud_type` and `fill_flag` are **integer codes**, not measurements — join to
  NREL's code tables to interpret them. `fill_flag != 0` flags a filled value.
- Timestamps are always fetched with `utc=true` (not configurable) so
  `valid_time` is UTC — NSRDB is the source of truth and the column is stored UTC.
- **Interval defaults to 60 min** (hourly). Note the phase: at 60 min the
  timestamps fall on the **half hour** (`00:30, 01:30, … 23:30`, the interval
  midpoint), *not* on the hour; 30-min data is stamped on the half hour from
  `00:00`.

## Caveats worth knowing

- **Modeled, not measured.** These are satellite + radiative-transfer estimates;
  treat them as a strong resource climatology, not point-accurate readings —
  especially under broken cloud, snow, or low sun.
- **Check `fill_flag`** before using a value for anything sensitive; nonzero
  means it was gap-filled.
- **Point snaps to a 4 km cell**, so nearby requests can return identical data.
- **One point / one year per CSV request** — `ingest_bronze` handles many points
  over a date range by looping `points × years` and trimming to the range (see
  [Fetching multiple points over a date range](#fetching-multiple-points-over-a-date-range)).
