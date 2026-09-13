# load pipeline

Lands raw bronze for the three sector load sources, and stacks the two dsgrid
industrial files into one metadata table and one timeseries table. This doc
describes the pipeline — what it produces, the decisions baked into it, the
budgets, and how failures are handled. For *what each source is*, read its own
directory: [`resstock/`](resstock/README.md), [`comstock/`](comstock/README.md),
[`dsgrid/`](dsgrid/README.md).

## What it does

| Module | Role |
|---|---|
| `resstock/bronze.py` | residential bronze — per-PUMA metadata + per-building PUMA timeseries |
| `comstock/bronze.py` | commercial bronze — per-PUMA metadata + per-building PUMA timeseries |
| `dsgrid/bronze.py` | industrial bronze — both dsgrid source files |
| `oedi_building_stock.py` | fetch / request / PUMA-assembly machinery shared by the two building-stock sources |
| `dsg_common.py` | `.dsg` HDF5 fetch + reconstruction machinery behind `dsgrid/` |
| `schema.py` | `LoadGeography` (one PUMA), `LoadGeographies` (the PUMAs a run is for), a fetch config per source, `IndustrialLoadRequestArgs` (the silver key), and the sources' time-axis facts |
| `coverage.py` | "do I already have this?" — the manifest read that makes a re-run cheap |
| `failures.py` | permanent-vs-transient classification, and the record of buildings a release cannot serve |
| `retry.py` | `Retry-After`-aware backoff, shared by both download helpers |
| `silver.py` | derives every bronze manifest key, and stacks both dsgrid sources by kind |
| `catalog.py` | registers all ten dataset types (8 bronze + 2 silver) in one call |

Orchestration lives in
[`src/batch_jobs/load_pipeline/flows.py`](../../../../../src/batch_jobs/load_pipeline):
five Prefect flows (`ingest_resstock`, `ingest_comstock`, `ingest_dsgrid`,
`industrial_load_silver`, `run_load_pipeline`). The package itself imports no
Prefect.

## The storage layout

Ten datasets, none of them partitioned into directories. The manifest is the
index — the same choice the climate pipeline makes per point, for the same reason:
a write already names one PUMA (or one state), so a `state=/puma=` directory would
hold a single file and repeat what the key says.

| dataset | one entry per |
|---|---|
| `resstock_metadata_bronze` | PUMA |
| `resstock_timeseries_bronze` | PUMA |
| `comstock_puma_metadata_bronze` | PUMA |
| `comstock_timeseries_bronze` | PUMA |
| `dsgrid_industrial_metadata_bronze` / `..._timeseries_bronze` | state |
| `dsgrid_industrial_gaps_metadata_bronze` / `..._gaps_timeseries_bronze` | state |
| `dsgrid_industrial_metadata_silver` / `..._timeseries_silver` | state |

Each building-stock write is keyed by the PUMA and nothing about what the fetch
returned, so it is a key a reader can build without knowing the building count in
advance. The key carried that count for a while, so a PUMA written short over a
building the release cannot serve declared itself in the manifest — but it made
every read a scan rather than a resolve. A shortfall is now reported where it is
durable anyway: in the rows, and in the refusals recorded on the run's flow
manifest.

Per-PUMA keying is what makes the pipeline **incremental** (adding a PUMA fetches
one PUMA), **resumable** (a re-run pays only for what is missing) and safe to run
in overlapping groups.

**ResStock's metadata is where key and source disagree, deliberately.** OEDI
publishes it one file per *state*, so the fetch is hoisted above the PUMA loop: a
run reads that 54 MB file once and cuts a write for each of the state's PUMAs it
was asked for. The trade is that a PUMA ingested in a later run re-downloads the
state file to cut its own slice — the price of every dataset being addressable by
the thing a caller actually has.

**dsgrid is the exception, and cannot be otherwise.** It publishes at county grain
with no PUMA column, and PUMAs do not nest inside counties — DC's single county
`G1100010` contains five of them. Splitting a county's industrial load across them
needs an allocation basis dsgrid does not provide, and copying it into each would
make the industrial total unrecoverable by summing: five times DC's real load. So
its geography stays `county_gisjoin`, the finest that is actually true of the data.
A reader holding a PUMA resolves its counties first.

The timeseries used to be written as 50-building chunks, which bought reuse and
resumability from one mechanism. One file per PUMA costs both: an interruption
re-fetches the PUMA whole, and the table must be assembled in memory to be written
(1.2 GB for a full ComStock PUMA, with a process high-water mark several times
that). Legacy chunk writes are still readable — a key carrying `first_bldg_id` is a
chunk, and a whole-PUMA write supersedes the chunks it came from rather than being
concatenated with them.

## The silver tables

Four bronze tables become two silver ones, stacked **by kind** rather than by
source:

```
dsgrid_industrial_metadata_bronze        ┐
dsgrid_industrial_gaps_metadata_bronze   ┘→ dsgrid_industrial_metadata_silver     (27 × 20 for DC)

dsgrid_industrial_timeseries_bronze      ┐
dsgrid_industrial_gaps_timeseries_bronze ┘→ dsgrid_industrial_timeseries_silver  (237,168 × 18 for DC)
```

It is a **union, not a horizontal join.** The two sources describe different
sectors of the same counties at different granularity — 4-digit NAICS subsectors
versus 2-digit sectors — so there is no key to join on. Joining on county alone
would fan each gaps row across all 26 subsectors and multiply its energy.

Two columns make the stack work:

| Column | Purpose |
|---|---|
| `naics_code` | harmonised sector id — `naics_subsector` (4 digits) or `naics_sector` (2 digits). They cannot collide. |
| `source` | which file the row came from, so manufacturing and non-manufacturing stay separable |

**The 12 end-use columns are null on non-manufacturing rows.** The gaps file
publishes a single un-decomposed end use, so there is nothing to put there, and a
null says that honestly where a zero would lie.

`electricity_total_mwh` and `annual_electricity_total_mwh` are populated on *every*
row, which is what makes the total recoverable:

```python
ts.group_by("timestamp", "county_gisjoin").agg(pl.col("electricity_total_mwh").sum())
```

gives the whole industrial load, both halves together — 249.749 GWh for DC,
matching the sum of the two bronze sources exactly.

**The axis stays dsgrid's own** — hourly, the 2012 modelled year, one national
clock (−05:00) for every county, timestamps still interval-*ending* as bronze
publishes them. This is a conforming of bronze, not a re-basing of it;
`schema.DSGRID_UTC_OFFSET_MINUTES` and `schema.DSGRID_YEAR` are what a consumer
needs to align it against the building stock later.

## Facts about the sources

Combining these with the building stock later means reconciling three things, none
of which this silver table does:

1. **Clock.** dsgrid publishes every county in the country on one national EST
   clock; ResStock and ComStock are on each state's own standard time. Both are
   DST-free, so a single offset resolves it — but it is 3 hours in California and
   exactly zero in DC, which is why a DC-only test cannot catch a mistake here.
2. **Calendar year.** dsgrid models 2012, the building stock 2018. Matching by
   nominal `(month, day, hour)` lines up seasonality and drops Feb 29. The residue
   is weekday alignment: 2012-01-01 is a Sunday and 2018-01-01 a Monday, so
   **Jan 1 – Feb 28 lands one weekday off** while Mar 1 onward aligns exactly.
3. **Interval.** dsgrid is hourly, the building stock 15-minute. An hourly average
   power held flat across its four sub-intervals is energy-preserving, but it means
   industrial contributes no sub-hourly structure.

Weights are a building-stock concern only: both building-stock timeseries tables
carry **unweighted per-model** kWh and need the metadata `weight` applied, whereas
dsgrid is published at real-world county scale already.

The release also publishes pre-weighted `timeseries_aggregates`, and this pipeline
deliberately does **not** ingest them — they carry no `bldg_id`, so they answer
"what does this PUMA draw" but never "which buildings drew it", and an aggregate
cannot be un-summed. Keeping bronze uniformly unweighted means one rule for
reading it: multiply by `weight` and sum.

Note ComStock's `weight` is scoped to the PUMA its metadata file covers — a
building represents stock across many PUMAs, and the file carries only this one's
share. That is the right quantity for PUMA-grain work and the wrong one to sum
across PUMAs.

## Running it

`run_load_pipeline` runs the three bronze ingests then the industrial silver
stack; the four narrower flows run one step each. Everything lands under one
`root_uri`.

A run takes **one or more PUMAs** (`LoadGeographies`, e.g. `{"pumas":
["G11000101", "G11000105"]}`), which is a run-level grouping only: each PUMA is
ingested as its own keyed writes, so the same PUMA asked for alone and asked for in
a group resolves to the same dataset — and a group that overlaps an earlier run
pays only for the PUMAs it adds. The PUMAs need not share a state; dsgrid bronze
and the industrial silver collapse them to their distinct states, since both are
published per state.

The two building-stock ingests fetch **one file per building** — a real PUMA is
~200 ResStock and ~950 ComStock buildings, so budget a few thousand HTTP requests,
times the number of PUMAs. A location means **all** of its buildings; there is
deliberately no cap, because a partial slice would write under the same params as a
complete PUMA and nothing downstream could tell an understated aggregate from a
real one. The building count is logged before the fan-out starts, which is the
signal an operator watches; a caller who genuinely wants a subset resolves the ids
and calls `fetch_puma_timeseries_table` directly.

How many PUMAs one *run* may carry is bounded by `schema.MAX_PUMAS_PER_RUN` (20),
because a flow's timeout is fixed when the module is imported — a longer list is
refused outright rather than quietly shortened, and splitting it across runs costs
nothing.

### Budgets

OEDI is public and publishes **no rate limit**, so there is no quota to encode.
The per-building fan-out is metered by two **Prefect global concurrency limits on
the server**, not in code, so they can be retuned without a deploy:

```bash
prefect global-concurrency-limit create oedi-api  --limit 28
prefect global-concurrency-limit create oedi-rate --limit 600 --slot-decay-per-second 28
```

A limit that does not exist is a **no-op**, which is what keeps the offline tests
server-free. `oedi-api` has a floor: the residential metadata fetch asks for 8 slots
at once (it is weighted by size — see below), and a request for more slots than the
limit holds is never satisfiable, so setting `oedi-api` below 8 does not slow the
run down, it stalls it.

| Limit | Bounds | Where the number comes from |
|---|---|---|
| `oedi-api` | requests **in flight** | **Self-imposed, measured.** A ResStock building file is ~6.3 MB, so 28 in flight already draws ~15.7 MB/s (~126 Mbit/s) and the link saturates. 48 in flight doubled mean fetch latency (10.8 s → 18.2 s) for 9% more throughput. |
| `oedi-rate` | how fast requests **start** | Without it, 20 fast responses start 20 more immediately, so a burst is bounded only by latency. The 600-slot pool lets a PUMA's buildings burst, then settles to the decay rate. |

These protect the run's own latency, not the provider: 2,224 fetches at 28
concurrent — and a run at 56 — produced **zero** retries, and 28 requests per ~10 s
is under 3/second against the 5,500 GET/s per prefix S3 documents.

Two host-shaped knobs are read from the environment at import (see `deploy.py`):

| Env var | Default | What it sizes |
|---|---|---|
| `LOAD_PIPELINE_OEDI_IN_FLIGHT` | 28 | the task pool, against `oedi-api`. **Raising the server limit alone does nothing** — a mapped child needs a worker before it can ask for an HTTP slot, so the smaller of the two wins. |
| `LOAD_PIPELINE_PUMAS_IN_FLIGHT` | 1 | how many PUMAs are fetched at once. A **memory** bound: ~1.7 GB resident per PUMA for both sources, roughly double at the concat. |

On a bandwidth-bound host raising either buys almost nothing — measured locally,
the link sits idle 2.4 s out of a 720 s fetch window (0.33%). On a host where the
link is not the ceiling (a VM in the bucket's own region) both are worth raising
together.

## Failure handling

A PUMA is hundreds of per-building files and some are genuinely absent: a release
lists a building whose timeseries was never published. Losing an hour-long ingest
to one of them is the failure mode the design avoids.

| Kind | Example | What happens |
|---|---|---|
| **Permanent** | OEDI 403 / 404 / 400 | Retries skipped, building recorded, the PUMA is written short |
| **Transient** | timeout, 429, 5xx, anything unrecognised | Retried — `download_object` internally, then the task twice |
| **Absent PUMA** | a GISJOIN no release has a file for | PUMA recorded and skipped, the rest of the run proceeds |

Classification happens **at the raise site**, not by inspecting messages — OEDI is
a plain public object store, so a status is enough. (The climate pipeline has to
match on message phrases because NSRDB and CDS report real rejections and their own
backend failures under the same status.) Everything unrecognised is **transient by
default**: a needlessly retried transient costs seconds, a wrongly permanent one
drops a building until someone passes `force_refresh`.

**A permanent failure is recorded, not fatal.** Records land on the run's flow
manifest under `metadata.failed_buildings`, and later runs read them back through
`failures.known_dead` and exclude those buildings before fetching. Without that a
PUMA holding one unpublished building re-requests that file on every run, forever,
and never settles. They are recorded on the failure path too, so a run that dies
partway still contributes what it learned.

**A transient failure fails the whole PUMA** rather than writing it short. That
data is expected to arrive, and a short write over it would make a blip
indistinguishable from a permanent hole — which no later run would look at again.

**The metadata phase fans out.** Both metadata fetches are submitted together
rather than walked, so a run's opening is one round-trip's worth of latency however
many PUMAs it covers — it used to be one per state plus one per PUMA, in series,
with nothing able to start behind them. They are metered on the same `oedi-api`
budget as every other request, weighted by size (a ~50 MB ResStock state file counts
as several ~6.3 MB building files), so fanning out cannot stampede the link the whole
design treats as its ceiling.

What stays ordered is the one real dependency: a PUMA's timeseries reads that PUMA's
metadata bronze for its building ids, so the phase still completes before the
timeseries fetches begin. How many PUMAs then fetch at once is
`LOAD_PIPELINE_PUMAS_IN_FLIGHT` — a memory bound, sized from the host (see
`flows.py`), not a politeness control.

**A PUMA the release does not have costs its own PUMA, not the run.** A GISJOIN can
be well formed and still name nothing: a code mistyped into a run form, or a
2020-vintage code against these 2010-vintage releases (a PUMA split since 2010 has
sub-codes, so e.g. NY `G36000400` does not exist — `G36000401/2/3` do). The metadata
phase is where that shows: ResStock finds no such rows in the state file, ComStock
gets a 404 on a per-PUMA object it never published. Either way the code is recorded
under `metadata.rejected_pumas`, dropped from the phases below, and **the run
continues for the PUMAs that are real**. A run whose every code was bad still fails —
there is no work left to save, and completing would report success for a run that
wrote nothing.

Rejection is **per source**: ResStock and ComStock are separate releases with
separately published geography, so a code one lacks is no evidence about the other,
and only that source's half is dropped. The state-grained steps (dsgrid bronze, the
industrial silver) keep a state as long as one of its PUMAs was real; a state reached
only by a bad code is dropped with it.

Unlike `failed_buildings`, a rejection is **never read back to skip a PUMA**. A
building the release refuses is a fact about the release, worth remembering so a PUMA
can settle. A PUMA it refuses is nearly always a typo — remembering it would mean the
corrected run got skipped by the record of its own mistake, and re-checking costs
only the metadata read the run was doing anyway.

**A cancelled run still closes its record.** Prefect cancels by sending SIGTERM,
which its engine raises as a `TerminationSignal` — a `BaseException`, so the
`except Exception` that records a failure never sees it. Every flow therefore closes
in a `finally`: the manifest is stamped `cancelled` (a terminal status, so it gets an
`end_time`) and keeps whatever the run had already discovered. Without it an
interrupted run was left reading `running` forever, in a process that had exited,
with its `failed_buildings` and `rejected_pumas` lost in memory.

**A short PUMA is the number to worry about.** Load data is summed and peaked, so a
PUMA holding 943 of its 946 buildings understates demand by ~0.3%, and nothing in
the data, the schema or the row count reveals it. `check_complete` compares what a
write holds against what its metadata lists and warns; `require_complete` raises
instead, for a caller that would rather fail than publish a number it cannot stand
behind.

## Reading what a run did

The flow says what each PUMA adds up to, and warns only when a re-run would help —
a PUMA short by exactly the buildings already recorded unavailable is as complete
as the release allows:

```
resstock_timeseries_bronze for PUMA G11000101: 197 building(s), complete
comstock_timeseries_bronze for PUMA G11000101: 943 of 946 building(s); the other 3
  are recorded unavailable, so this PUMA is as complete as the release allows.
```

Buildings recorded dead across all runs, without opening any flow manifest by hand:

```python
from external_data.load_pipeline import failures

for source in ("resstock", "comstock"):
    print(source, sorted(failures.known_dead(ROOT, source, "G11000101")))
```

No Prefect needed — the record is a package-level contract, so reading it does not
require the orchestration layer.

## Caveats worth knowing

- **The PUMA id is the location, by design — not a placeholder for lat/long.** All
  three sources are census-geography-keyed and none understands a coordinate, and
  the state derives from the GISJOIN's FIPS prefix, so one field addresses every
  file the pipeline reads. A coordinate door would unlock no data and would add a
  resolution step, plus a boundary-geometry ingest, in front of a working key. A
  caller holding a coordinate resolves it once, elsewhere, and should not
  approximate it by centroid or county — that spends a whole ingest on the wrong
  buildings and returns a plausible number. The industrial silver needs even less:
  dsgrid is per state, so `IndustrialLoadRequestArgs` takes only a state code.
- **The weather years differ.** A hot 2018 August afternoon in ComStock has no
  counterpart in dsgrid's 2012. Least harmful for industrial, which is driven by
  shift schedules rather than temperature, and the two weather-sensitive sectors do
  share 2018.
- **Both dsgrid files make up the industrial total**, and the stack keeps them in
  one table, so summing `electricity_total_mwh` gives the whole thing. Filtering
  to `source == 'industrial'` gives manufacturing only — for DC that omits 35% of
  the load, since its non-manufacturing half is construction-heavy.
- **A PUMA in a state's minority time zone needs its offset spelled out.**
  `STATE_STANDARD_UTC_OFFSET_MINUTES` holds each state's *dominant* zone; for the
  13 states in `MULTI_ZONE_STATES`, pass `local_standard_utc_offset_minutes` on
  that PUMA rather than accepting an hour's error.

## Test fixtures

`packages/external-data/tests/load_pipeline/*/fixtures` holds ~500 KB of **real**
OEDI slices for DC (`state=DC`, PUMA `G11000101`, county `G1100010`): 3 real
buildings per building-stock source over 2018-01-01…01-08, and the full DC dsgrid
reconstruction (26 subsectors + 1 gaps sector) over 2012-01-01…01-08, i.e. 168
hourly interval starts. `silver/fixtures` holds the bronze tables the silver tests
stack; the per-source `fixtures` directories hold the raw `.dsg` and parquet slices
the bronze tests read through the real parsers. Three buildings is not a whole
PUMA, so the absolute MW there is real but *partial*.

## Still open

| # | Item | Why it matters |
|---|---|---|
| 1 | **The legacy chunk read path is still carried.** `read_puma_timeseries` matches on a subset of the key so 50-building chunk writes still resolve. It can go once none remain on disk, which would make a PUMA read a plain `resolve_manifest`. | Cleanup |
| 2 | **ComStock metadata is downloaded whole for 14 columns.** The per-PUMA file carries ~1,300; the curated set is 14. The strongest case in the pipeline for pushing the projection to the source. | Bandwidth |
| 3 | **A PUMA ingested in a later run re-downloads its state's ResStock metadata** to cut its own slice (54 MB for a large state). Accepted so every dataset is addressable by a PUMA; a state-level metadata cache would remove it. | Known, deliberate |
