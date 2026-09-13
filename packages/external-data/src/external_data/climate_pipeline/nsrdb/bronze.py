"""
NSRDB GOES Aggregated (PSM v4.0.0) timeseries -- bronze ingestion.

Fetches NSRDB solar timeseries from the NREL API and writes **one single-file
bronze dataset per point**, each with its own manifest entry. No physical
partitions -- the manifest is the spatial index (see
``climate_pipeline.point_manifest``), which lets a re-run skip points it already
has and lets concurrent fetches write without contending.

Two request types from ``climate_pipeline.schema`` play distinct roles:

- :class:`~external_data.climate_pipeline.schema.ClimatePipelineRequestArgs`
  (points + date range) is the manifest key, narrowed to a single point per
  write, and wrapped in an ``NsrdbDatasetKey`` that adds the interval.
- :class:`~external_data.climate_pipeline.schema.NsrdbRequestArgs` is the NSRDB
  *fetch config* (interval, attributes), combined with the joined request to
  build each per-(point, year) query.

NSRDB serves one point and one calendar year per request, so a request spanning
several points/years fans out to ``points x years`` HTTP calls. The *fetch* unit
is therefore a point-year, while the *manifest* unit is a point: each point's
years are combined and trimmed to the requested range before its single write.

**Attributes are a selection, not a fixture.** A request may name any non-empty
subset of the curated set and the ingest runs on it: the CSV is checked and
projected against what was *asked for*, and the curated columns left out are
written as nulls so the stored schema never changes. The selection joins the
interval on the write's manifest key, which is what stops a narrow write reading
as coverage for a later, wider request -- see :func:`coverage_match`.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import logging
import os
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import httpx
import patito as pt
import polars as pl

import common.exceptions
from common.frames import BaseDataFrameSchema
from common.storage import columnar
from common.storage.manifest import ManifestRow
from external_data.climate_pipeline import failures, point_manifest, schema

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

BASE_URL = (
    "https://developer.nlr.gov/api/nsrdb/v2/solar/nsrdb-GOES-aggregated-v4-0-0-download"
)

# NSRDB dataset id comes from the shared schema.
DATASET = schema.NSRDB_SOURCE_DATASET

DATASET_NAME = "nsrdb_solar_bronze"

DEFAULT_WRITER = "nsrdb_bronze"

DEFAULT_BASE_DIR = "bronze"

DEFAULT_TIMEOUT_SECONDS = 120

# Cap Retry-After so a hostile/huge header can't stall a batch indefinitely
_MAX_RETRY_AFTER_SECONDS = 120.0

# The published budget per API key: 1,000 requests/hour on a rolling window, 429
# once exceeded (https://developer.nlr.gov/docs/rate-limits/). No per-second limit
# is documented, so the hourly figure is the real ceiling.
DOCUMENTED_REQUESTS_PER_HOUR = 1_000
# Every response carries the budget actually applied, which may differ from the
# default if the key was granted more. Read it rather than assume -- a drifted
# assumption is invisible until the 429s start.
_RATE_LIMIT_HEADER = "X-RateLimit-Limit"
_RATE_REMAINING_HEADER = "X-RateLimit-Remaining"
# Warn while there is still room to react, not once the budget is gone.
_LOW_BUDGET_FRACTION = 0.1

KEY_COLUMNS = ["valid_time", "latitude", "longitude"]

# Curated attributes become the bronze value columns, one for one and in the same
# order -- NSRDB names an attribute exactly as bronze names its column.
#
# This is the *stored* shape, not the fetched one: a request may ask for a subset
# (see :func:`selected_columns`), and the columns it leaves out are written as
# nulls. Bronze therefore has one schema whatever the selection -- which is what
# lets ``read_bronze_uris`` hand files from runs with different selections to a
# single ``pl.read_parquet``, and lets silver keep reading the columns it needs.
VALUE_COLUMNS = list(schema.NSRDB_ATTRIBUTES)

_TIME_COLUMNS = ["Year", "Month", "Day", "Hour", "Minute"]

# Integer-coded categorical value columns
_INT_VALUE_COLUMNS = ["cloud_type", "fill_flag"]

_RAW_TO_VALUE_COLUMN = {
    "GHI": "ghi",
    "DNI": "dni",
    "DHI": "dhi",
    "Clearsky GHI": "clearsky_ghi",
    "Clearsky DNI": "clearsky_dni",
    "Clearsky DHI": "clearsky_dhi",
    "Cloud Type": "cloud_type",
    "Fill Flag": "fill_flag",
    "Temperature": "air_temperature",
    "Wind Speed": "wind_speed",
    "Surface Albedo": "surface_albedo",
    "Solar Zenith Angle": "solar_zenith_angle",
    "Relative Humidity": "relative_humidity",
    "Pressure": "surface_pressure",
}

# Point coordinates are echoed once in the CSV metadata header
_LOCATION_DECIMALS = 4


# --------------------------------------------------------------------------- #
# Credentials
# --------------------------------------------------------------------------- #


def _rc_path() -> Path:
    """
    Location of the optional credentials file
    """
    return Path(os.environ.get("NSRDBRC", os.path.expanduser("~/.nsrdbrc")))


def _read_rc(path: Path) -> dict[str, str]:
    """
    Parse a ``~/.nsrdbrc`` file into a ``{name: value}`` dict
    """
    config: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            name, value = line.split(":", 1)
            config[name.strip()] = value.strip()
    return config


def _rc_value(name: str) -> str:
    """Read a single value from the optional ``~/.nsrdbrc`` file (empty if absent)."""
    rc_path = _rc_path()
    return _read_rc(rc_path).get(name, "") if rc_path.exists() else ""


def _api_key() -> str:
    """Resolve the NREL API key from env, then the rc file; raise if neither has it."""
    api_key = os.environ.get("NSRDB_API_KEY", "") or _rc_value("key")
    if not api_key:
        raise common.exceptions.PipelineValueError(
            "no NSRDB API key found; set NSRDB_API_KEY or add a `key:` line to "
            f"{_rc_path()} (get a free key at https://developer.nlr.gov/signup/)"
        )
    return api_key


def _email() -> str:
    """Resolve the NREL contact email from env, then the rc file; raise if neither."""
    email = os.environ.get("NSRDB_EMAIL", "") or _rc_value("email")
    if not email:
        raise common.exceptions.PipelineValueError(
            "no NSRDB contact email found; set NSRDB_EMAIL or add an `email:` "
            f"line to {_rc_path()} (required by the NSRDB API)"
        )
    return email


# --------------------------------------------------------------------------- #
# Request
# --------------------------------------------------------------------------- #


def selected_columns(nsrdb: schema.NsrdbRequestArgs | None) -> list[str]:
    """
    The value columns a fetch for *nsrdb* actually returns, in stored order.

    What the CSV is checked and projected against. The rest of
    :data:`VALUE_COLUMNS` is null in the write -- the stored schema is fixed, the
    fetched subset is not.
    """
    attributes = schema.NSRDB_ATTRIBUTES if nsrdb is None else nsrdb.attributes
    selected = set(attributes)
    return [column for column in VALUE_COLUMNS if column in selected]


def build_query(
    nsrdb: schema.NsrdbRequestArgs,
    latitude: float,
    longitude: float,
    year: int,
) -> dict[str, str]:
    """
    Build the NSRDB query params for a single point + year: attributes/interval
    from the NSRDB fetch config, the point and year supplied per request.
    ``utc=true`` is always requested -- NSRDB is the source of truth and
    ``valid_time`` is stored as UTC.
    """
    return {
        "wkt": f"POINT({longitude} {latitude})",
        "attributes": ",".join(nsrdb.attributes),
        "names": str(year),
        "interval": str(nsrdb.interval),
        "utc": "true",
        "leap_day": "true",
    }


# --------------------------------------------------------------------------- #
# Dataset schema (catalog / manifest)
# --------------------------------------------------------------------------- #


class NsrdbBronzeSchema(BaseDataFrameSchema):
    """
    One row per timestamp of the NSRDB point timeseries
    """

    valid_time: dt.datetime = pt.Field(
        dtype=pl.Datetime(time_unit="us", time_zone="UTC"),
        description="observation timestamp (UTC)",
    )
    latitude: float = pt.Field(dtype=pl.Float64, description="grid-cell latitude")
    longitude: float = pt.Field(dtype=pl.Float64, description="grid-cell longitude")

    ghi: float | None = pt.Field(
        dtype=pl.Float32, description="global horizontal irradiance (W/m^2)"
    )
    dni: float | None = pt.Field(
        dtype=pl.Float32, description="direct normal irradiance (W/m^2)"
    )
    dhi: float | None = pt.Field(
        dtype=pl.Float32, description="diffuse horizontal irradiance (W/m^2)"
    )
    clearsky_ghi: float | None = pt.Field(
        dtype=pl.Float32, description="clear-sky global horizontal irradiance (W/m^2)"
    )
    clearsky_dni: float | None = pt.Field(
        dtype=pl.Float32, description="clear-sky direct normal irradiance (W/m^2)"
    )
    clearsky_dhi: float | None = pt.Field(
        dtype=pl.Float32, description="clear-sky diffuse horizontal irradiance (W/m^2)"
    )
    cloud_type: int | None = pt.Field(
        dtype=pl.Int32, description="NSRDB cloud-type code"
    )
    fill_flag: int | None = pt.Field(dtype=pl.Int32, description="NSRDB fill-flag code")
    air_temperature: float | None = pt.Field(
        dtype=pl.Float32, description="2m air temperature (C)"
    )
    wind_speed: float | None = pt.Field(
        dtype=pl.Float32, description="wind speed (m/s)"
    )
    surface_albedo: float | None = pt.Field(
        dtype=pl.Float32, description="surface albedo (fraction)"
    )
    solar_zenith_angle: float | None = pt.Field(
        dtype=pl.Float32, description="solar zenith angle (degrees)"
    )
    relative_humidity: float | None = pt.Field(
        dtype=pl.Float32, description="relative humidity (%)"
    )
    surface_pressure: float | None = pt.Field(
        dtype=pl.Float32, description="surface pressure (mbar)"
    )


# --------------------------------------------------------------------------- #
# Fetch (raw)
# --------------------------------------------------------------------------- #


def _new_client() -> httpx.Client:
    """Default HTTP client for NSRDB downloads"""
    return httpx.Client(timeout=DEFAULT_TIMEOUT_SECONDS, follow_redirects=True)


def _retry_wait(
    response: httpx.Response, backoff_seconds: float, attempt: int
) -> float:
    """
    Seconds to wait before the next retry: honor a ``Retry-After`` header (secs)
    when the server sends one, else exponential backoff. Capped so a large header
    can't stall a batch.
    """
    retry_after = response.headers.get("Retry-After")
    if retry_after is not None:
        try:
            return min(float(retry_after), _MAX_RETRY_AFTER_SECONDS)
        except ValueError:
            pass  # non-integer (HTTP-date) form -- fall back to backoff
    return backoff_seconds * (2 ** (attempt - 1))


def observed_rate_budget(response: httpx.Response) -> tuple[int, int] | None:
    """
    The ``(limit, remaining)`` the server reports, or ``None`` if it reports none.

    Grounds the budget in what the provider is applying rather than what this code
    assumes. A key can be granted more or less than
    :data:`DOCUMENTED_REQUESTS_PER_HOUR`, and the headers are the only place the
    real number is visible.
    """
    try:
        limit = int(response.headers[_RATE_LIMIT_HEADER])
        remaining = int(response.headers[_RATE_REMAINING_HEADER])
    except KeyError, TypeError, ValueError:
        return None
    return limit, remaining


def _log_rate_budget(response: httpx.Response) -> None:
    """Report the server's own view of the remaining budget, loudly when it is low."""
    budget = observed_rate_budget(response)
    if budget is None:
        return
    limit, remaining = budget
    if limit != DOCUMENTED_REQUESTS_PER_HOUR:
        logger.info(
            "NSRDB reports a budget of %d requests/hour, not the documented %d; "
            "size backfills against the reported figure",
            limit,
            DOCUMENTED_REQUESTS_PER_HOUR,
        )
    if limit > 0 and remaining <= limit * _LOW_BUDGET_FRACTION:
        logger.warning(
            "NSRDB budget nearly spent: %d of %d requests left this window; "
            "further requests will be throttled with 429s",
            remaining,
            limit,
        )
    else:
        logger.debug("NSRDB budget: %d of %d requests left", remaining, limit)


def download_csv(
    request: dict[str, str],
    *,
    api_key: str | None = None,
    email: str | None = None,
    max_retries: int = 3,
    backoff_seconds: float = 2.0,
    client: httpx.Client | None = None,
) -> str:
    """
    Make a real request and return the raw CSV response body as a string
    """
    # Resolve each credential independently so a missing email doesn't raise
    # about the key (or vice versa), and a caller-supplied value is never re-looked-up.
    if api_key is None:
        api_key = _api_key()
    if email is None:
        email = _email()

    params = {
        **request,
        "api_key": api_key,
        "email": email,
        # NREL asks who is requesting; these are labels, not credentials.
        "full_name": os.environ.get("NSRDB_FULL_NAME", "energy-pipelines user"),
        "affiliation": os.environ.get("NSRDB_AFFILIATION", "energy-pipelines"),
    }
    url = f"{BASE_URL}.csv"

    owns_client = client is None

    client = client if client is not None else _new_client()
    last_exc: Exception | None = None
    try:
        for attempt in range(1, max_retries + 1):
            logger.info(
                "requesting NSRDB aggregated CSV (wkt=%s, year=%s, interval=%smin, "
                "attempt=%d/%d)",
                request.get("wkt"),
                request.get("names"),
                request.get("interval"),
                attempt,
                max_retries,
            )
            try:
                response = client.get(url, params=params)
            except httpx.RequestError as exc:
                last_exc = exc
                if attempt < max_retries:
                    wait = backoff_seconds * (2 ** (attempt - 1))
                    logger.warning("request failed (%s); retrying in %.1fs", exc, wait)
                    time.sleep(wait)
                continue

            # 429 is transient like 5xx, and expected: the batch multiplies
            # request count by points x years.
            if response.status_code == 429 or response.status_code >= 500:
                # A 429 carries the budget just exceeded -- the moment to report it.
                _log_rate_budget(response)
                last_exc = httpx.HTTPStatusError(
                    f"HTTP {response.status_code}",
                    request=response.request,
                    response=response,
                )
                if attempt < max_retries:
                    wait = _retry_wait(response, backoff_seconds, attempt)
                    logger.warning(
                        "HTTP %d; retrying in %.1fs", response.status_code, wait
                    )
                    time.sleep(wait)
                continue

            if 400 <= response.status_code < 500:
                body = response.text.strip()
                permanent = failures.as_permanent_nsrdb(response.status_code, body)
                if permanent is not None:
                    raise permanent
                # An unrecognised 4xx is NSRDB's backend, not our request -- it
                # reports its own failures as 400 too. Retry like a 5xx rather
                # than dead-letter a point that would succeed next attempt.
                last_exc = httpx.HTTPStatusError(
                    f"HTTP {response.status_code}: {body}",
                    request=response.request,
                    response=response,
                )
                if attempt < max_retries:
                    wait = _retry_wait(response, backoff_seconds, attempt)
                    logger.warning(
                        "HTTP %d (%s); retrying in %.1fs",
                        response.status_code,
                        body[:120],
                        wait,
                    )
                    time.sleep(wait)
                continue

            _log_rate_budget(response)
            return response.text
    finally:
        if owns_client:
            client.close()

    raise common.exceptions.PipelineError(
        f"failed to download NSRDB data after {max_retries} attempts"
    ) from last_exc


# --------------------------------------------------------------------------- #
# Parse -> bronze table
# --------------------------------------------------------------------------- #


def _read_metadata(text: str) -> dict[str, str]:
    """
    Parse the two-row NSRDB metadata header (field names + values) into a dict
    """
    meta = pl.read_csv(io.BytesIO(text.encode()), n_rows=1, infer_schema_length=0)
    return {col: meta[col][0] for col in meta.columns}


def csv_to_bronze_table(
    text: str, expected: Sequence[str] | None = None
) -> pl.DataFrame:
    """
    Parse the raw NSRDB CSV into a typed one-row-per-timestamp bronze frame.

    *expected* is the value columns this download asked for -- the whole curated
    set by default. Checking against what was requested rather than against
    :data:`VALUE_COLUMNS` is what lets a subset fetch through: an attribute nobody
    asked for is absent, not missing.
    """
    metadata = _read_metadata(text)

    # Scan every row when inferring dtypes: columns like "Wind Speed" are whole
    # numbers for the first rows and fractional later, which trips the default
    # 100-row inference into i64
    df = pl.read_csv(io.BytesIO(text.encode()), skip_rows=2, infer_schema_length=None)

    missing_time = [c for c in _TIME_COLUMNS if c not in df.columns]
    if missing_time:
        msg = (
            f"NSRDB CSV is missing timestamp column(s) {missing_time}; "
            f"got columns {sorted(df.columns)}"
        )
        raise common.exceptions.PipelineValueError(msg)

    df = df.rename({k: v for k, v in _RAW_TO_VALUE_COLUMN.items() if k in df.columns})

    # Fail loud if the download is missing an attribute it *asked for*
    wanted = list(VALUE_COLUMNS if expected is None else expected)
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        msg = (
            f"NSRDB download is missing expected variable(s) {missing}; "
            f"got columns {sorted(df.columns)}"
        )
        raise common.exceptions.PipelineValueError(msg)

    # Round to grid precision so the location partition values are exact
    latitude = round(float(metadata["Latitude"]), _LOCATION_DECIMALS)
    longitude = round(float(metadata["Longitude"]), _LOCATION_DECIMALS)

    int_cols = [c for c in wanted if c in _INT_VALUE_COLUMNS]
    float_cols = [c for c in wanted if c not in _INT_VALUE_COLUMNS]
    return (
        df.with_columns(
            pl.datetime(
                pl.col("Year"),
                pl.col("Month"),
                pl.col("Day"),
                pl.col("Hour"),
                pl.col("Minute"),
            )
            .dt.replace_time_zone("UTC")
            .alias("valid_time"),
            pl.lit(latitude, dtype=pl.Float64).alias("latitude"),
            pl.lit(longitude, dtype=pl.Float64).alias("longitude"),
            *[pl.col(c).cast(pl.Int32) for c in int_cols],
            *[pl.col(c).cast(pl.Float32) for c in float_cols],
        )
        .select(KEY_COLUMNS + wanted)
        .sort("valid_time")
    )


# --------------------------------------------------------------------------- #
# Write -> permanent storage
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #


def _fetch_point_year_table(
    latitude: float,
    longitude: float,
    year: int,
    nsrdb: schema.NsrdbRequestArgs,
    client: httpx.Client | None = None,
) -> pl.DataFrame:
    """
    Fetch and parse a single (point, year) NSRDB timeseries into a bronze frame,
    using the interval/attributes from the NSRDB fetch config
    """
    request = build_query(nsrdb, latitude, longitude, year)
    return csv_to_bronze_table(
        download_csv(request, client=client), selected_columns(nsrdb)
    )


def fetch_point_year_table(
    point: tuple[float, float],
    year: int,
    nsrdb: schema.NsrdbRequestArgs | None = None,
    client: httpx.Client | None = None,
) -> pl.DataFrame:
    """
    Fetch one (point, year) as a bronze frame.

    The unit an orchestrator fans out over -- a *point-year*, because the CSV
    endpoint serves one of each per request and ``names=2018,2019`` is rejected
    with ``400 Invalid value(s)``. Free of orchestration imports.
    """
    nsrdb = nsrdb if nsrdb is not None else schema.NsrdbRequestArgs()
    return _fetch_point_year_table(point[0], point[1], year, nsrdb, client=client)


def requested_years(request: schema.ClimatePipelineRequestArgs) -> list[int]:
    """
    The calendar years a request spans, in order -- one HTTP request each.

    One source for the year list, so the fetch loop, unit expansion and shortfall
    check cannot disagree about how many requests a point costs.
    """
    return list(range(request.start_date.year, request.end_date.year + 1))


def narrow_to_fetched(
    request: schema.ClimatePipelineRequestArgs, last_year: int
) -> schema.ClimatePipelineRequestArgs:
    """
    The request narrowed to end with *last_year*.

    A manifest entry records **one** range, so a point missing its later years
    must claim only the span it holds. Claiming the full range would be worse than
    writing nothing: :func:`covers` would report the hole as covered, so later runs
    skip the point and silver joins over data that is not there.
    """
    end = min(request.end_date, dt.date(last_year, 12, 31))
    return request.model_copy(update={"end_date": end})


def request_units(
    request: schema.ClimatePipelineRequestArgs,
) -> list[tuple[tuple[float, float], int]]:
    """
    Every (point, year) a request expands to, exact-duplicate points removed.

    One entry per HTTP request, so the sequential and fanned-out paths agree.
    """
    seen: set[tuple[float, float]] = set()
    points: list[tuple[float, float]] = []
    for latitude, longitude in request.points:
        if (latitude, longitude) not in seen:
            seen.add((latitude, longitude))
            points.append((latitude, longitude))
    return [(point, year) for point in points for year in requested_years(request)]


def combine_point_tables(
    tables: Sequence[pl.DataFrame],
    request: schema.ClimatePipelineRequestArgs,
    n_requests: int,
) -> pl.DataFrame:
    """
    Concatenate per-(point, year) frames, trim to the requested days, dedupe.

    Split from :func:`ingest_bronze` so a concurrent caller reuses the same trim
    and guard logic.
    """
    combined = pl.concat(tables)

    # NSRDB returns whole calendar years; trim to the requested day range.
    combined = combined.filter(
        pl.col("valid_time").dt.date().is_between(request.start_date, request.end_date)
    )

    # The trim can discard every row; fail loud rather than writing a
    # successful-looking manifest row pointing at nothing.
    if combined.is_empty():
        msg = (
            f"no NSRDB rows fell inside [{request.start_date}, {request.end_date}] "
            f"after trimming {n_requests} request(s)"
        )
        raise common.exceptions.PipelineValueError(msg)

    before = combined.height
    combined = combined.unique(subset=KEY_COLUMNS).sort(KEY_COLUMNS)
    if combined.height < before:
        logger.warning(
            "bronze: dropped %d duplicate key row(s)", before - combined.height
        )
    return combined


def point_params(
    point: tuple[float, float],
    request: schema.ClimatePipelineRequestArgs,
    nsrdb: schema.NsrdbRequestArgs,
) -> schema.NsrdbPointSliceKey:
    """
    The manifest identity of one point's slice of a batch request.

    ``interval`` is part of the key: 30- and 60-minute fetches of one point are
    different datasets, not two writes racing for one entry. Flat alongside the
    point, so a reader reaches it without unwrapping a level.

    ``attributes`` rides along for the same reason, since a write no longer
    implies the whole curated set -- the stored columns cannot tell the two apart,
    so the key has to.
    """
    return schema.NsrdbPointSliceKey(
        point=point_manifest.normalise(point),
        start_date=request.start_date,
        end_date=request.end_date,
        interval=nsrdb.interval,
        attributes=nsrdb.attributes,
    )


def stored_shape(table: pl.DataFrame) -> pl.DataFrame:
    """
    The table in the shape bronze stores: every curated column, in order.

    An attribute the fetch did not ask for is added as an all-null column rather
    than left out, so every write has the same columns whatever was selected.
    Two reasons that matters more than the few bytes it costs: ``read_bronze_uris``
    hands several points' files to one ``pl.read_parquet``, which needs them to
    agree on a schema; and ``NsrdbBronzeSchema`` -- validated on the way out --
    declares all of them. Which attributes a write *holds* is recorded in its
    manifest key (``NsrdbPointSliceKey.attributes``), not inferred from its
    columns. The null keeps the column's declared type, so the two integer-coded
    columns do not come back as floats.
    """
    absent = [c for c in VALUE_COLUMNS if c not in table.columns]
    if not absent:
        return table.select(KEY_COLUMNS + VALUE_COLUMNS)
    return table.with_columns(
        *[
            pl.lit(
                None, dtype=pl.Int32 if c in _INT_VALUE_COLUMNS else pl.Float32
            ).alias(c)
            for c in absent
        ]
    ).select(KEY_COLUMNS + VALUE_COLUMNS)


def write_point_bronze(
    table: pl.DataFrame,
    point: tuple[float, float],
    request: schema.ClimatePipelineRequestArgs,
    nsrdb: schema.NsrdbRequestArgs,
    root_uri: str | Path,
    writer: str = DEFAULT_WRITER,
    write_time: dt.datetime | None = None,
) -> ManifestRow:
    """
    Write one point (all its years, already combined) as its own dataset.

    The *fetch* unit is a point-year, because that is all the API serves; the
    *write* unit is a point, so one manifest entry answers "do I have this
    point for this range?" in a single lookup.

    Padded to the stored shape first, so a subset fetch writes the same columns a
    full one does -- see :func:`stored_shape`.
    """
    return columnar.write_dataset(
        stored_shape(table),
        NsrdbBronzeSchema,
        DATASET_NAME,
        point_params(point, request, nsrdb),
        str(root_uri),
        writer=writer,
        write_time=write_time,
    )


def stored_attributes(row: ManifestRow) -> frozenset[str]:
    """
    The NSRDB attributes a manifest entry holds values for.

    An entry written before the selection was recorded has no ``attributes``, and
    is read as the full curated set -- correctly, since a subset could not be
    requested then.
    """
    params = json.loads(row.params_json)
    return frozenset(params.get("attributes", schema.NSRDB_ATTRIBUTES))


def coverage_match(
    nsrdb: schema.NsrdbRequestArgs,
) -> Callable[[ManifestRow], bool]:
    """
    A :func:`point_manifest.split_by_coverage` filter accepting only an entry that
    matches this fetch's ``interval`` **and** holds every attribute it asks for.

    Both halves of NSRDB's identity in one predicate, because a planner needs
    both: the interval decides whether the entry is even the same dataset (a
    30-minute write is not coverage for a 60-minute request), and the attributes
    decide whether it holds what was asked for.

    Attributes are a superset test, not equality: an entry with the full curated
    set answers a two-attribute request, so narrowing a selection re-fetches
    nothing. Widening one does re-fetch, which is the point -- without this the
    earlier, narrower write would read as coverage, the point would never be
    fetched again, and the added attributes would stay null in bronze and in every
    silver built from it.
    """
    wanted = frozenset(nsrdb.attributes)

    def _matches(row: ManifestRow) -> bool:
        params = json.loads(row.params_json)
        if params.get("interval") != nsrdb.interval:
            return False
        return wanted <= stored_attributes(row)

    return _matches


def resolve_point_uris(
    request: schema.ClimatePipelineRequestArgs,
    nsrdb: schema.NsrdbRequestArgs,
    root_uri: str | Path,
    as_of: dt.datetime | None = None,
    *,
    allow_missing: bool = False,
) -> dict[point_manifest.Point, str]:
    """
    The parquet URI holding each requested point's bronze.

    Only entries at this ``interval`` match, since it is part of the identity.
    Split from :func:`read_points_bronze` like the ERA5 one: a node-by-node builder
    resolves every point in a single scan before looping.
    """
    rows, missing = point_manifest.covered_rows(
        request.points,
        DATASET_NAME,
        str(root_uri),
        point_manifest.read_point_key,
        request.start_date,
        request.end_date,
        as_of=as_of,
        match=point_manifest.interval_match(nsrdb.interval),
    )
    if missing:
        detail = (
            f"{len(missing)} of {len(missing) + len(rows)} requested point(s) for "
            f"[{request.start_date}, {request.end_date}] at "
            f"{nsrdb.interval}-minute interval under {root_uri}: "
            f"{point_manifest.describe_points(missing)}"
        )
        # Strict by default; see the ERA5 reader for why ``allow_missing`` exists.
        if not allow_missing or not rows:
            raise common.exceptions.PipelineValueError(
                f"no NSRDB bronze covering {detail}; ingest the bronze slice "
                "before building silver"
            )
        logger.warning("NSRDB bronze is missing %s; continuing without them", detail)
    return {point: row.data_uri for point, row in rows.items()}


def read_bronze_uris(uris: Iterable[str]) -> pl.DataFrame:
    """Read already-resolved per-point bronze files into one deduped frame."""
    frame = pl.read_parquet(list(uris))
    return frame.unique(subset=KEY_COLUMNS).sort(KEY_COLUMNS)


def read_points_bronze(
    request: schema.ClimatePipelineRequestArgs,
    nsrdb: schema.NsrdbRequestArgs,
    root_uri: str | Path,
    as_of: dt.datetime | None = None,
    *,
    allow_missing: bool = False,
) -> pl.DataFrame:
    """
    Read every requested point's bronze back as one frame.

    Only entries written at this ``interval`` match, since the interval is part
    of the identity.
    """
    uris = resolve_point_uris(
        request, nsrdb, root_uri, as_of, allow_missing=allow_missing
    )
    return read_bronze_uris(uris.values())


def ingest_bronze(
    request: schema.ClimatePipelineRequestArgs,
    nsrdb: schema.NsrdbRequestArgs | None = None,
    root_uri: str | Path = DEFAULT_BASE_DIR,
    writer: str = DEFAULT_WRITER,
    client: httpx.Client | None = None,
    write_time: dt.datetime | None = None,
) -> list[ManifestRow]:
    """
    Fetch the NSRDB timeseries for every point in ``request`` across the calendar
    years its date range spans, and write **one bronze dataset per point** (its
    years combined and trimmed to the range), returning a manifest row for each.
    Each is keyed by an ``NsrdbDatasetKey`` -- the point and range plus
    ``nsrdb.interval``, so 30- and 60-minute fetches stay distinct. ``nsrdb`` is
    the fetch config (interval/attributes); the attributes default to the curated
    set, and any non-empty subset of it fetches just those -- the rest are stored
    as nulls and the selection is recorded on each entry.

    Per point, not per batch, so what this writes is what
    :func:`resolve_point_uris` (and therefore silver) can find -- a batch entry
    cannot answer "do I have this point?" and is invisible to every reader.

    One point and one year per request, so this issues
    ``len(unique points) x len(spanned years)`` calls -- mind the rate limit on
    large batches. A shared HTTP client is reused across them.

    A **permanently** unavailable year ends that point's range rather than
    discarding it: the years already fetched are written under a key narrowed by
    :func:`narrow_to_fetched`. A point whose *first* year is unavailable has no
    honest range to claim and is skipped. Matches the mapped flow, so both paths
    produce the same entries for the same failures.

    Fetches those units **sequentially**. An orchestrator that wants them
    concurrent should map :func:`fetch_point_year_table` over
    :func:`request_units` and pass each point's results to
    :func:`combine_point_tables` + :func:`write_point_bronze` -- the same three
    steps this function performs, which is why they are public.
    """
    if nsrdb is None:
        nsrdb = schema.NsrdbRequestArgs()

    if "://" not in str(root_uri):
        root_uri = str(Path(root_uri).resolve())

    units = request_units(request)
    # The API unit is a point-year, the *manifest* unit a point: group years back
    # together and write one entry per point, the only shape readers resolve.
    years_by_point: dict[tuple[float, float], list[int]] = {}
    for point, year in units:
        years_by_point.setdefault(point, []).append(year)

    owns_client = client is None
    client = client if client is not None else _new_client()
    rows: list[ManifestRow] = []
    skipped: list[tuple[float, float]] = []
    shortened: list[tuple[float, float]] = []
    index = 0
    try:
        for point, years in years_by_point.items():
            tables: list[pl.DataFrame] = []
            narrowed = request
            try:
                for year in years:
                    index += 1
                    table = fetch_point_year_table(point, year, nsrdb, client=client)
                    logger.info(
                        "bronze: request %d/%d (%.4f, %.4f) year %d -> %d row(s)",
                        index,
                        len(units),
                        point[0],
                        point[1],
                        year,
                        len(table),
                    )
                    tables.append(table)
            except failures.PermanentFetchError as exc:
                # Only *permanent* failures are absorbed; a transient one raises,
                # since that data is expected and recording the gap as final would
                # stop anything looking for it.
                if not tables:
                    # Nothing salvageable: no honest range to claim.
                    logger.warning(
                        "bronze: (%.4f, %.4f) permanently unavailable, skipping: %s",
                        point[0],
                        point[1],
                        exc,
                    )
                    skipped.append(point)
                    continue
                # Keep the years already fetched: those requests are spent, and
                # narrowing makes a partial set distinguishable from a complete one.
                narrowed = narrow_to_fetched(request, years[len(tables) - 1])
                logger.warning(
                    "bronze: (%.4f, %.4f) %d permanently unavailable (%s); keeping "
                    "the %d year(s) already fetched and ending this point at %s",
                    point[0],
                    point[1],
                    years[len(tables)],
                    exc,
                    len(tables),
                    narrowed.end_date,
                )
                shortened.append(point)
            combined = combine_point_tables(tables, narrowed, len(tables))
            rows.append(
                write_point_bronze(
                    combined,
                    point,
                    narrowed,
                    nsrdb,
                    root_uri,
                    writer=writer,
                    write_time=write_time,
                )
            )
    finally:
        if owns_client:
            client.close()

    if not rows:
        msg = (
            f"no NSRDB bronze produced: all {len(years_by_point)} point(s) are "
            "permanently unavailable"
        )
        raise common.exceptions.PipelineValueError(msg)

    logger.info(
        "bronze: wrote %d point(s) from %d request unit(s), one entry each; "
        "skipped %d permanently unavailable, narrowed %d to the years available",
        len(rows),
        len(units),
        len(skipped),
        len(shortened),
    )
    return rows


if __name__ == "__main__":
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    _demo = schema.ClimatePipelineRequestArgs(
        points=((37.7749, -122.4194),),
        start_date=dt.date(2023, 6, 1),
        end_date=dt.date(2023, 6, 2),
    )
    for _row in ingest_bronze(_demo):
        print(_row.data_uri)
