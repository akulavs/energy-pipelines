# climate pipeline

Lands per-point bronze from two weather sources and joins them into an hourly
silver grid. This doc describes the pipeline — what it produces, the decisions
baked into it, the provider budgets, and how failures are handled. For *what each
source is*, read its own directory: [`era5/`](era5/README.md),
[`nsrdb/`](nsrdb/README.md).

## What it does

| Module | Role |
|---|---|
| `era5/bronze.py` | ERA5-Land hourly weather — one CDS request per 0.1° grid node |
| `era5/land_mask.py` | ERA5-Land's own land-sea mask, to refuse sea points before fetching |
| `nsrdb/bronze.py` | NSRDB solar — one NREL request per (point, year) |
| `silver.py` | the ERA5 × NSRDB join, built and written node by node |
| `point_manifest.py` | per-point manifest lookups shared by all three |
| `failures.py` | permanent-vs-transient classification |
| `dead_letter.py` | the recorded-failure format, read back so later runs skip dead points |
| `schema.py` | request models and the per-point manifest keys |
| `catalog.py` | registers the three dataset types |

Orchestration lives in `src/batch_jobs/climate_pipeline/flows.py`: four Prefect
flows (`ingest_era5`, `ingest_nsrdb`, `era5_nsrdb_silver`, `run_climate_pipeline`).
The package itself imports no Prefect.

## The storage layout

Three datasets, all keyed **per point**:

```
era5_weather_bronze     one entry per 0.1° grid node
nsrdb_solar_bronze      one entry per point (+ interval in the key)
era5_nsrdb_silver       one entry per 0.1° grid node
```

There are **no physical partitions**. The manifest is the spatial index: a scan
returns every point ever written, and the reader filters in memory. Each entry is
an immutable, UUID-stamped write, which is what makes concurrent per-point fetches
safe with no storage-layer changes.

A key names a single point, not a collection of one:

```json
era5 / silver  {"point":[37.4,-122.1],"start_date":"2020-01-01","end_date":"2024-12-31"}
nsrdb          {"point":[37.77,-122.42],"start_date":"2023-01-01","end_date":"2023-12-31","interval":60}
```

A multi-point request produces N entries, one per point. `read_point_key` still
accepts the older `{"points":[[…]]}` shape so writes made before the rename
resolve; that branch can go once none remain.

Per-point keying is what makes the pipeline **incremental** (adding 50 points
fetches 50), **resumable** (a killed backfill re-runs only what is missing) and
**concurrent** (no shared entry to contend on).

## The silver join

ERA5-anchored, built one grid node at a time:

- Every ERA5 node/hour is kept. NSRDB's finer 4 km cells are snapped onto the
  0.1° grid, nearest kept, and joined as a **left** join on `(node, hour)`.
- A node with no NSRDB — outside coverage, or nothing snapped to it — keeps its
  weather with `None` solar. That is a legitimate state, not an error.
- Grouping is by **node**, not by requested point: two points in one cell would
  otherwise emit the same `(node, hour)` rows twice.

Node-by-node is the memory bound. Thousands of points over ~27 hourly years is
hundreds of millions of rows; looping holds one node at a time, with no streaming
machinery.

## Provider budgets

Held as **Prefect global concurrency limits on the server**, not in code, so they
can be retuned without a deploy. Create them once per environment:

```bash
prefect global-concurrency-limit create cds-api    --limit 5
prefect global-concurrency-limit create nsrdb-api  --limit 18
prefect global-concurrency-limit create nsrdb-rate --limit 1000 --slot-decay-per-second 0.27
```

A limit that does not exist is a **no-op** — the code runs unthrottled and nothing
errors, which is also what keeps the offline tests server-free.

| Limit | Value | Where the number comes from |
|---|---|---|
| `cds-api` | 5 concurrent | **Self-imposed.** Copernicus publishes no fixed per-user concurrency figure and states its limits vary with system workload, so there is no number to match. A deliberately small politeness cap. |
| `nsrdb-api` | 18 concurrent | **Self-imposed.** NSRDB documents no concurrent-request limit. With the hourly budget doing the real throttling, this is thread-pool backpressure. |
| `nsrdb-rate` | 1,000 slots, 0.27/s decay | **Documented default.** NSRDB publishes 1,000 requests/hour per key on a rolling window ([docs](https://developer.nlr.gov/docs/rate-limits/)), 429 once exceeded. 0.27/s ≈ 972/hour, just under the ceiling; the 1,000-slot burst is what a rolling window allows. A key granted more should be re-derived — see below. |

Both providers rate-limit **per account**, so more machines buy no throughput —
concurrency in one process already reaches the ceiling.

**`--limit` is burst, not rate.** On a rate limit the sustained throughput comes
entirely from `--slot-decay-per-second`; `--limit` only caps how many requests can
leave back to back. Raising `--limit` alone changes nothing about the rate. Derive
both from the key's hourly budget:

```
--limit <hourly budget>  --slot-decay-per-second <hourly budget / 3600>
```

**Check the real budget, don't assume it.** Every NSRDB response carries
`X-RateLimit-Limit` / `X-RateLimit-Remaining`; `observed_rate_budget` reads them
and the fetch logs them, warning at 10% remaining and noting when the server
reports a different limit than the documented default. A key granted a higher
quota is invisible otherwise.

The `create` block above stays at the documented default, which is what a fresh
key gets. This environment's key reports **10,000/hour**, so its limit is set to
`--limit 10000 --slot-decay-per-second 2.7778`. That figure is key-specific: read
`X-RateLimit-Limit` and re-derive rather than copying it.

Thread pools are sized *above* the limits (`ERA5_MAX_WORKERS = 8`,
`NSRDB_MAX_WORKERS = 24`) so the limit is the bottleneck, never the pool.

### Flow timeouts

Sized from the NSRDB rate, which binds: one request per `(point, year)` at
1,000/hour, so wall clock ≈ `(points × years) / 1000` hours.

| Flow | Timeout | Covers |
|---|---|---|
| `ingest_era5`, `ingest_nsrdb` | 18 h | ~650 points of full history |
| `era5_nsrdb_silver` | 4 h | local work; scales with nodes, not budget |
| `run_climate_pipeline` | 22 h | the larger bronze phase + silver |

A thousands-of-points full-history backfill does **not** fit one run at the
documented quota (2,000 points ≈ 54 h), and no timeout would change that. Split
it across runs — the manifest skip makes each subsequent run fetch only what is
missing. `_warn_if_over_budget` says so at the start rather than a day in.

These are sized against the **documented floor**, not against whatever a given key
is granted, so they hold in any environment. A larger quota finishes
proportionally sooner: at 10,000/hour the same 18 h covers ~6,600 points and that
2,000-point backfill takes ~5.4 h. The timeouts stay put — too short cancels a
legitimate run partway, too long only delays noticing a hung one.

`_warn_if_over_budget` estimates from the documented rate too, so on a
higher-quota key it can warn that a run will not fit when it comfortably would.
That is why it warns rather than raises.

## Failure handling

A scattered request always contains points a provider cannot serve. Losing a
multi-hour run to one of them is the failure mode the design avoids.

| Kind | Example | What happens |
|---|---|---|
| **Permanent** | NSRDB `No data available at the provided location` / `Invalid value(s)` / 401 / 403 / 404; CDS "not produced any data"; ERA5-Land returns no values | Retries skipped, point recorded, run continues |
| **Transient** | timeout, 429, 5xx, an unrecognised 4xx, anything else | Retried — `download_csv` internally, then the task twice |

Everything unrecognised is **transient by default**: a needlessly retried
transient costs seconds, a wrongly permanent one loses a point until someone
passes `force_refresh`.

NSRDB reports its *own* backend failures as 400 as well, so the status cannot
classify on its own — `{"errors":["Data processing failure."]}` comes back for
requests that succeed unchanged on retry. That phrase also accompanies the
genuinely permanent responses, so the discriminator is the phrase naming the
location or the parameters, not the generic one.

**Dead-lettering.** Permanent failures are recorded in the run's flow manifest
under `metadata.failed_points`, at `{root_uri}/_flows/{flow_id}.json`. Later runs
read them back and skip those points rather than re-asking a provider that has
already refused. `force_refresh` bypasses the list — the way back for a point
judged dead in error.

**Sea points cost nothing.** `land_mask.is_sea` checks ERA5-Land's own `lsm`
field before any CDS call. Only certain sea is refused (`lsm == 0`, 64% of the
globe); cells with any land fraction are still fetched, with the all-values-missing
guard as backstop. The mask is never a gate — if it cannot be obtained, lookups
abstain and the fetch proceeds.

**Partial NSRDB years.** A permanently unavailable year ends that point's range
rather than discarding it: the years already fetched are written under a key
narrowed to what they cover. A transient year failure fails the whole point
instead, so a blip is never recorded as a permanent hole.

**Gaps in silver.** Strict by default — a missing point stops the build, since
the usual cause is a bronze step never run. A gap an earlier run recorded as
permanent is built over automatically. `allow_missing_points` forces the
permissive behaviour for *any* gap.

## Reading what a run did

Plan logs name the points and the range, not just counts:

```
ERA5  2023-06-01..2025-06-01: fetching 3 [(37.7, -122.4), (37.8, -122.4), (37.9, -122.4)]
NSRDB 2023-06-01..2024-12-31: fetching 0; 2 already covered [(37.7, -122.4) 2023-06-01..2024-12-31, …]; 1 recorded dead [(0.0, -140.0)]
```

The NSRDB header shows the **clamped** window — the range actually fetched when a
request runs past NSRDB's coverage.

Dead points across all runs, without opening any flow:

```python
from external_data.climate_pipeline import dead_letter, point_manifest
from external_data.climate_pipeline.era5 import bronze as era5

for source, grain in (("era5", era5.node), ("nsrdb", point_manifest.normalise)):
    for point, why in sorted(dead_letter.known_points(ROOT, source, grain).items()):
        print(source, point, why)
```

No Prefect needed — the record is a package-level contract, so reading it does not
require the orchestration layer.

## Still open

| # | Item | Why it matters |
|---|---|---|
| 1 | **A newer narrower write shadows an older wider one.** `existing_points` resolves by recency, so re-requesting a wide range after a narrow write refetches even though the wide data is on disk. Safe — an entry never claims a span it lacks — and pinned by test. | Known, deliberate |
