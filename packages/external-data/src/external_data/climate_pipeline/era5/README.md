# ERA5-Land

The data source behind `external_data.climate_pipeline.era5`. This doc describes
the dataset — what it is, where it comes from, what the values mean, and how to
use them. For *how we ingest it*, read `bronze.py` in this directory; for how the
pipeline around it works, the [pipeline README](../README.md).

## What it is

**ERA5-Land** is a global, hourly, land-surface climate **reanalysis** produced
by [ECMWF](https://www.ecmwf.int/en/era5-land) for the **Copernicus Climate
Change Service (C3S)** on behalf of the European Commission. It provides a
gap-free, physically consistent record of the land surface (temperature, soil,
snow, wind, radiation, precipitation) from **January 1950 to ~2–3 months before
present**.

- Overview: <https://www.ecmwf.int/en/era5-land>
- Full docs: <https://confluence.ecmwf.int/spaces/CKB/pages/140385202/ERA5-Land+data+documentation>
- Reference paper: Muñoz-Sabater et al. (2021), *ESSD* — <https://essd.copernicus.org/articles/13/4349/2021/>

## Where we pull it from

The Copernicus Climate Data Store (CDS). We use the **timeseries** variant,
which returns a per-point hourly series rather than gridded fields:

- Dataset id: `reanalysis-era5-land-timeseries`
- CDS page: <https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-timeseries>
- Gridded sibling (for reference): <https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land>

Access is via the CDS API (the `cdsapi` client); see **Credentials** below.

## Licence

**CC-BY-4.0** — free to use, redistribute and build products on, **commercial use
included**, provided the source is credited. Read from the CDS catalogue, so it
can be re-checked in one request:

```
GET https://cds.climate.copernicus.eu/api/catalogue/v1/collections/reanalysis-era5-land-timeseries
    -> "license": "CC-BY-4.0"        (checked 2026-08-11)
```

### Attribution

CC-BY does not prescribe wording. The conventional Copernicus forms are:

| our data is | use |
|---|---|
| unmodified | *Generated using Copernicus Climate Change Service information [year]* |
| modified | *Contains modified Copernicus Climate Change Service information [year]* |

Anything downstream of [the silver join](../silver.py) is the **modified** case —
it converts Kelvin to °C, wind components to speed and direction, and metres to
millimetres, and drops columns. `[year]` is the year of the data, not the year it
was downloaded.

### Which instrument you accept

The catalogue's `license` field is a machine-readable summary. The CDS licence
vocabulary also carries a separate **`licence-to-use-copernicus-products`**
(rev 12) alongside plain `cc-by`, and no public endpoint states which of the two
this dataset gates on — the acceptance step described under **Credentials** is
where you see it. For anything contractual, read what the dataset page actually
asks you to accept rather than relying on the summary above.

Neither the European Commission nor ECMWF accepts responsibility for how the data
is used.

## Credentials

The CDS API authenticates with two values — there is no username/password sent
per request:

- **`url`** — the CDS API endpoint: `https://cds.climate.copernicus.eu/api`.
- **`key`** — a **Personal Access Token (PAT)** tied to your CDS account. Create
  a free account and copy the token from your profile:
  <https://cds.climate.copernicus.eu/profile>. (The client also accepts the
  legacy `<UID>:<APIKEY>` form, but new accounts get a token-only PAT.)

You must also **accept each dataset's licence once**, on its CDS dataset page —
until you do, the API returns `403` for that dataset even with a valid token.

### How the client finds them

Our code calls `cdsapi.Client()` with no arguments, so it uses cdsapi's default
credential resolution, in this order:

1. **Environment variables** — `CDSAPI_URL` and `CDSAPI_KEY`.
2. **RC file** — `~/.cdsapirc` (override the path with the `CDSAPI_RC` env var).

RC file format:

```
url: https://cds.climate.copernicus.eu/api
key: <personal-access-token>
```

Equivalent environment-variable form:

```
export CDSAPI_URL=https://cds.climate.copernicus.eu/api
export CDSAPI_KEY=<personal-access-token>
```

## How the source produces it (reanalysis)

A **reanalysis** blends a physical model with historical observations to
reconstruct the past state of the Earth system on a regular grid — complete
even where no instrument ever measured.

ERA5-Land specifically **replays the land component of the parent ERA5
reanalysis**:

1. ERA5 assimilates worldwide observations (satellites, stations, radiosondes)
   into an atmospheric model via 4D-Var. This is where real-world observations
   enter the picture.
2. Those ERA5 atmospheric fields (temperature, humidity, wind, radiation,
   precipitation) are downscaled and used to **force** ECMWF's land-surface
   model, **H-TESSEL**, run at higher resolution.
3. ERA5-Land itself runs **without data assimilation** — it is a single
   land-only simulation. Observations influence it only *indirectly*, through
   the ERA5 forcing. This is what makes it cheap enough to extend to near-real
   time.

**Resolution & cadence**

| | |
|---|---|
| Spatial | ~9 km native; published on a **0.1° × 0.1°** lat/lon grid |
| Temporal | **Hourly** |
| Coverage | Global **land** (no open ocean), 1950 → present |
| Latency | Consolidated data lags real time by ~2–3 months |

## The values

One row per hour for a single 0.1° (~9 km) grid point. All 13 variables are
stored **as delivered — no unit conversion**. Keys: `valid_time`, `latitude`,
`longitude`.

A fetch may ask for **any non-empty subset** of the 13 (`Era5RequestArgs.variables`);
an unrecognised name is rejected. The stored table always has all 13 columns —
the ones not fetched are null — so bronze has one schema whatever a run selected,
and the selection is recorded on the write's manifest entry rather than inferred
from its columns. A later run asking for more re-fetches the point; asking for
fewer reuses what is there. A subset propagates into silver as null columns (the
join reads bronze as it finds it), so drop a variable only when nothing
downstream needs it.

| Column | Variable | Units | What it is / how it's used |
|---|---|---|---|
| `t2m`  | 2 m air temperature | K | Primary demand driver — heating/cooling load. |
| `d2m`  | 2 m dewpoint temperature | K | With `t2m` → humidity / wet-bulb → AC latent load. |
| `ssrd` | Surface solar radiation downwards | J m⁻² | Solar PV potential. Hourly total; ÷3600 → average W m⁻². |
| `u10`  | 10 m wind, U component | m s⁻¹ | With `v10` → wind speed for wind generation (10 m → hub height). |
| `v10`  | 10 m wind, V component | m s⁻¹ | See `u10`. |
| `stl1` | Soil temperature, level 1 (0–7 cm) | K | Shallow ground temp. |
| `stl2` | Soil temperature, level 2 (7–28 cm) | K | Subsurface thermal context. |
| `stl3` | Soil temperature, level 3 (28–100 cm) | K | Buried-cable ampacity (dynamic cable rating). |
| `stl4` | Soil temperature, level 4 (100–289 cm) | K | Deep soil temp; deeper cable/asset rating. |
| `skt`  | Skin (surface) temperature | K | Equipment/surface temp proxy; transformer & line ratings. |
| `sde`  | Snow depth | m | PV soiling, line loading, cold-load pickup. |
| `snowc`| Snow cover | % | Fraction of cell under snow. |
| `tp`   | Total precipitation | m | Hydro inflow, storm/outage correlation. Hourly total. |

### Dropped in the silver join

The [silver join](../silver.py) carries most ERA5 variables through (converted to
friendly units — °C, m/s wind speed + direction, mm precip), but **drops**:

- `ssrd` — silver takes solar/irradiance from NSRDB (`ghi`/`dni`/`dhi`, etc.), so
  ERA5's surface solar radiation is not carried.
- `stl2`, `stl3`, `stl4` — only level-1 soil temperature (`stl1`) is kept; the
  deeper soil layers are dropped.

### Conventions to know

- **Temperatures are Kelvin** (`t2m`, `d2m`, `stl*`, `skt`). Subtract 273.15 for °C.
- **`ssrd` and `tp` are already hourly** in this timeseries product — each value
  is the total over that one hour, **not** a running accumulation, so **no
  de-accumulation is needed** (differencing consecutive hours would be wrong).
  This differs from the classic gridded ERA5-Land hourly product, whose
  `ssrd`/`tp` accumulate from 00 UTC. For `ssrd`, divide J m⁻² by 3600 for an
  average W m⁻².
- **`tp` and `sde` are in metres** (of water equivalent for precipitation).

## Caveats worth knowing

- **Not measurements.** These are model output constrained by observations, not
  station readings. Local extremes and sub-9 km features are smoothed.
- **Land only, and CDS does not say so.** Ask for a point at sea and the API
  returns the full time axis with every variable missing rather than an error. The
  ingest refuses such points up front against ERA5-Land's own land-sea mask (see
  `land_mask.py`) and rejects an all-missing response as a permanent failure, so
  neither silently becomes a row of nulls.
- **Do not de-accumulate** `ssrd`/`tp`. The timeseries API already returns them as
  hourly values (verified against the bronze fixtures); differencing them or
  treating them as running sums would corrupt the data.
- **Snow depth over the Antarctic/Greenland ice sheets is unreliable** (outdated
  glacier mask); values ≥10 m flag glacier grid cells, not real snowpack.
- **Near-real-time data is preliminary** and may be revised when consolidated.

## Capturing extreme events (future work)

Because ERA5-Land is a reanalysis averaged over a ~9 km grid cell, it **smooths
short-lived, localized extremes** — peak heat, cold snaps, wind gusts, and
intense convective rain are damped relative to what a nearby instrument actually
recorded.

A common mitigation is to **bias-correct the reanalysis against nearby
ground-station observations**: pull actual readings from a station close to the
point, align them with the ERA5-Land series over an overlapping period, and
correct for systematic offsets — and, ideally, the under-representation of
extremes.

**Candidate observation source — [Meteostat](https://dev.meteostat.net/python):**

- Open historical weather/climate data from **real stations**, aggregated from
  national weather services (NOAA, Germany's DWD, …).
- Python library returns Pandas frames (hourly / daily / monthly) and can find
  the nearest station(s) to a lat/lon point.
- **Free, no API key, no quota**; data under CC BY 4.0. Depends on `pandas` +
  `pyarrow`.

**Proposed future work:** cross-examine ERA5-Land against Meteostat station data
for the same location and period to quantify and correct biases — especially at
the extremes — before the reanalysis drives capacity/rating analysis. Caveat:
station records have their own gaps and siting effects, so treat them as a
reference to correct *toward*, not as absolute ground truth.
