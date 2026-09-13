"""
Request-args schemas for the climate pipeline.

Three plain request models:

1. :class:`ClimatePipelineRequestArgs` -- the user-facing **joined** request: a
   batch of ``(lat, lon)`` points over a date range. This is the bronze lookup
   key; it holds the *what/where/when*.
2. :class:`Era5RequestArgs` -- ERA5-Land **fetch config** (how to fetch), e.g.
   the variable set.
3. :class:`NsrdbRequestArgs` -- NSRDB **fetch config** (how to fetch), e.g. the
   interval and attribute set.

The source-specific models hold *only* what's specific to each source -- no
points or dates, so those are never duplicated. A real fetch combines the joined
request (points + dates) with a source's fetch config. Clamping NSRDB to its
coverage window is join logic and lives in ``silver.py``. The curated variable
sets, valid intervals, and NSRDB's coverage window below are the single source
of truth.

Both curated sets are menus: a request picks any non-empty subset of one and
nothing outside it. What is left out is stored as nulls rather than dropped, so
each bronze dataset keeps a single schema whatever a run selected, and the
selection is recorded on the write's manifest key.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated, Literal, Self

import annotated_types
import pydantic

import common.exceptions
import common.models

# --------------------------------------------------------------------------- #
# Shared coordinate types
# --------------------------------------------------------------------------- #

type Latitude = Annotated[float, annotated_types.Interval(ge=-90, le=90)]
type Longitude = Annotated[float, annotated_types.Interval(ge=-180, le=180)]
type Point = tuple[Latitude, Longitude]

# --------------------------------------------------------------------------- #
# ERA5-Land constants
# --------------------------------------------------------------------------- #

ERA5_SOURCE_DATASET = "reanalysis-era5-land-timeseries"

# Curated ERA5-Land variables requested as a point timeseries, each mapped to the
# short column name CDS/xarray returns it under. One mapping rather than two
# parallel lists: the fetch names the variable, every column downstream is the
# short name, and a subset request has to translate between them.
ERA5_VARIABLE_COLUMNS: dict[str, str] = {
    "2m_temperature": "t2m",
    "2m_dewpoint_temperature": "d2m",
    "surface_solar_radiation_downwards": "ssrd",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "soil_temperature_level_1": "stl1",
    "soil_temperature_level_2": "stl2",
    "soil_temperature_level_3": "stl3",
    "soil_temperature_level_4": "stl4",
    "skin_temperature": "skt",
    "snow_depth": "sde",
    "snow_cover": "snowc",
    "total_precipitation": "tp",
}

# The whole curated set, in the order CDS is asked for it -- which is not the
# order bronze stores the columns in (``era5.bronze.VALUE_COLUMNS``). Also the
# default a request that names no variables gets, and the order any subset is
# normalised to.
ERA5_VARIABLES: tuple[str, ...] = tuple(ERA5_VARIABLE_COLUMNS)

# --------------------------------------------------------------------------- #
# NSRDB constants
# --------------------------------------------------------------------------- #

NSRDB_SOURCE_DATASET = "nsrdb-GOES-aggregated-v4-0-0"

# Curated NSRDB attributes, in the order the columns are stored -- NSRDB names an
# attribute exactly as bronze names its column, so unlike ERA5 there is no second
# mapping to keep in step. A request picks any non-empty subset of these.
NSRDB_ATTRIBUTES: tuple[str, ...] = (
    "ghi",
    "dni",
    "dhi",
    "clearsky_ghi",
    "clearsky_dni",
    "clearsky_dhi",
    "cloud_type",
    "fill_flag",
    "air_temperature",
    "wind_speed",
    "surface_albedo",
    "solar_zenith_angle",
    "relative_humidity",
    "surface_pressure",
)

NSRDB_DEFAULT_INTERVAL = 60
# ERA5-Land is hourly, so the silver join reads the 60-minute NSRDB slice.
NSRDB_SILVER_INTERVAL = 60

# Coverage window the GOES Aggregated PSM v4 product serves. Single source of
# truth used (by silver.py) to clamp a joined request's NSRDB window; bump it
# here when NSRDB publishes a new year.
NSRDB_VALID_YEARS: tuple[int, ...] = tuple(range(1998, 2025))
NSRDB_MIN_DATE = dt.date(NSRDB_VALID_YEARS[0], 1, 1)
NSRDB_MAX_DATE = dt.date(NSRDB_VALID_YEARS[-1], 12, 31)

# --------------------------------------------------------------------------- #
# 1. Joined request (user-facing)
# --------------------------------------------------------------------------- #


class ClimatePipelineRequestArgs(common.models.FrozenModel, extra="forbid"):
    """
    Lat/long point(s) spanning a date range
    """

    points: tuple[Point, ...] = pydantic.Field(
        min_length=1,
        description=(
            "One or more points, each a two-item list in decimal degrees where "
            "the first item is the latitude (-90 to 90) and the second is the "
            "longitude (-180 to 180)."
        ),
    )
    start_date: dt.date = pydantic.Field(
        default=NSRDB_MIN_DATE,
        description=(
            "Start date. Default is the earliest date covered by both ERA5-Land "
            f"and NSRDB ({NSRDB_MIN_DATE})."
        ),
    )
    end_date: dt.date = pydantic.Field(
        default=NSRDB_MAX_DATE,
        description=(
            "End date. Default is the latest date covered by both ERA5-Land and "
            f"NSRDB ({NSRDB_MAX_DATE})."
        ),
    )

    @pydantic.model_validator(mode="after")
    def _check_range(self) -> Self:
        if self.start_date > self.end_date:
            msg = (
                f"start_date ({self.start_date}) must be on or before "
                f"end_date ({self.end_date})"
            )
            raise common.exceptions.PipelineValueError(msg)
        return self


# --------------------------------------------------------------------------- #
# 2. ERA5-Land request
# --------------------------------------------------------------------------- #


class Era5RequestArgs(common.models.FrozenModel, extra="forbid"):
    """
    ERA5 fetch
    """

    source_dataset: str = pydantic.Field(
        default=ERA5_SOURCE_DATASET,
        description="ERA5 dataset",
    )
    variables: tuple[str, ...] = pydantic.Field(
        default=ERA5_VARIABLES,
        min_length=1,
        description=(
            "ERA5 variables to fetch. Any non-empty subset of the curated set; "
            "the ones left out are stored as nulls, so the bronze schema does "
            "not change with the selection. Names must come from the curated "
            f"set: {', '.join(ERA5_VARIABLES)}."
        ),
    )

    @pydantic.field_validator("variables", mode="after")
    @classmethod
    def _curated_subset(cls, variables: tuple[str, ...]) -> tuple[str, ...]:
        """
        Accept any non-empty subset of the curated set, normalised to curated order.

        A *subset* is the point: dropping variables the caller does not need is
        the one knob on this fetch, and a CDS request is paid for per variable in
        queue and download time. An *unknown* name is still rejected, and up
        front -- before the slow CDS download -- because the bronze schema has no
        column to put it in, so fetching it could only end in a silent drop.

        Normalising (dedupe, curated order) makes two spellings of the same
        selection one request: same CDS payload, same ``request_hash``, same
        manifest params, so re-asking in a different order does not refetch.
        """
        unknown = [v for v in variables if v not in ERA5_VARIABLE_COLUMNS]
        if unknown:
            raise common.exceptions.PipelineValueError(
                f"Unknown ERA5 variable(s) {unknown}; choose from "
                f"{list(ERA5_VARIABLES)}"
            )
        selected = set(variables)
        return tuple(v for v in ERA5_VARIABLES if v in selected)


# --------------------------------------------------------------------------- #
# 3. NSRDB request
# --------------------------------------------------------------------------- #


class NsrdbRequestArgs(common.models.FrozenModel, extra="forbid"):
    """
    NSRDB fetch
    """

    source_dataset: str = pydantic.Field(
        default=NSRDB_SOURCE_DATASET,
        description="NSRDB dataset",
    )
    interval: Literal[30, 60] = pydantic.Field(
        default=NSRDB_DEFAULT_INTERVAL,
        description="Temporal resolution in minutes",
    )
    attributes: tuple[str, ...] = pydantic.Field(
        default=NSRDB_ATTRIBUTES,
        min_length=1,
        description=(
            "NSRDB attributes to fetch. Any non-empty subset of the curated set; "
            "the ones left out are stored as nulls, so the bronze schema does "
            "not change with the selection. Names must come from the curated "
            f"set: {', '.join(NSRDB_ATTRIBUTES)}."
        ),
    )

    @pydantic.field_validator("attributes", mode="after")
    @classmethod
    def _curated_subset(cls, attributes: tuple[str, ...]) -> tuple[str, ...]:
        """
        Accept any non-empty subset of the curated set, normalised to curated order.

        Mirrors :meth:`Era5RequestArgs._curated_subset`, for the same reasons: a
        subset is the knob worth having, and an unknown name is rejected up front
        because the bronze schema has no column to put it in. NSRDB bills the
        *request*, not the attribute, so a subset buys a narrower table rather
        than a cheaper call -- but the two sources behaving alike is worth more
        than the difference.

        Normalising (dedupe, curated order) makes two spellings of the same
        selection one request: same query string, same manifest params, so
        re-asking in a different order does not refetch.
        """
        unknown = [a for a in attributes if a not in NSRDB_ATTRIBUTES]
        if unknown:
            raise common.exceptions.PipelineValueError(
                f"Unknown NSRDB attribute(s) {unknown}; choose from "
                f"{list(NSRDB_ATTRIBUTES)}"
            )
        selected = set(attributes)
        return tuple(a for a in NSRDB_ATTRIBUTES if a in selected)


class NsrdbDatasetKey(common.models.FrozenModel, extra="forbid"):
    """
    Manifest identity of an NSRDB bronze slice: the joined request (points + date
    range) plus the fetch ``interval``. Including ``interval`` makes 30- and
    60-minute fetches of the same points/range **distinct** datasets rather than
    two writes colliding on one key. The silver join reads the 60-minute slice
    (:data:`NSRDB_SILVER_INTERVAL`) to match ERA5-Land's hourly grid.
    """

    request: ClimatePipelineRequestArgs = pydantic.Field(
        description="the joined request (points + date range) this slice covers",
    )
    interval: Literal[30, 60] = pydantic.Field(
        description="fetch interval in minutes that distinguishes this slice",
    )


# --------------------------------------------------------------------------- #
# 4. Per-point manifest keys
# --------------------------------------------------------------------------- #


class PointSliceKey(common.models.FrozenModel, extra="forbid"):
    """
    Manifest identity of **one point's** slice: the point itself, not a
    collection that happens to hold one.

    Distinct from :class:`ClimatePipelineRequestArgs` because the two answer
    different questions. A *request* is what a caller asked for and is
    legitimately multi-point. A *write* covers exactly one location. Keying the
    write with the request model forced a one-element tuple, which read badly in
    the manifest (``"points":[[37.4,-122.1]]``) and made every reader defend
    against a multi-point key that a per-point write can never legitimately have.
    """

    point: Point = pydantic.Field(
        description=(
            "The single point this write covers: a two-item list in decimal "
            "degrees, latitude (-90 to 90) then longitude (-180 to 180)."
        ),
    )
    start_date: dt.date = pydantic.Field(
        description="First day the write covers.",
    )
    end_date: dt.date = pydantic.Field(
        description="Last day the write covers.",
    )

    @pydantic.model_validator(mode="after")
    def _check_range(self) -> Self:
        if self.start_date > self.end_date:
            msg = (
                f"start_date ({self.start_date}) must be on or before "
                f"end_date ({self.end_date})"
            )
            raise common.exceptions.PipelineValueError(msg)
        return self


class Era5PointSliceKey(PointSliceKey):
    """
    A point slice plus the ERA5 ``variables`` the write actually holds.

    Recorded because a write no longer implies the whole curated set. The stored
    file always carries every curated column (the ones not fetched are null), so
    nothing in the *data* distinguishes a three-variable write from a full one --
    only this field does. Without it a subset write would read as coverage for
    any later request, and the point would never be fetched again: the columns it
    skipped would stay null forever, in bronze and in every silver built from it.

    Absent from an entry written before this field existed, which is why readers
    treat a missing ``variables`` as the full curated set -- back then a subset
    could not be requested.
    """

    variables: tuple[str, ...] = pydantic.Field(
        description="ERA5 variables this write holds values for.",
    )


class NsrdbPointSliceKey(PointSliceKey):
    """
    A point slice plus the fetch ``interval``, which is part of NSRDB's identity:
    30- and 60-minute fetches of one point are different datasets, not two writes
    racing for one key.

    ``attributes`` records what the write actually holds, for the same reason
    :class:`Era5PointSliceKey` carries ``variables``: the stored file always has
    every curated column (unfetched ones null), so nothing in the data separates a
    two-attribute write from a full one, and without the record a narrow write
    would read as coverage for every later request. An entry written before the
    field existed has none, and is read as the full curated set -- back then a
    subset could not be requested.

    Flat rather than nesting the slice under a ``request`` field -- the nesting
    only existed because the key reused the multi-point request model, and it made
    every reader unwrap a level to reach the point.
    """

    interval: Literal[30, 60] = pydantic.Field(
        description="Fetch interval in minutes that distinguishes this slice.",
    )
    attributes: tuple[str, ...] = pydantic.Field(
        description="NSRDB attributes this write holds values for.",
    )
