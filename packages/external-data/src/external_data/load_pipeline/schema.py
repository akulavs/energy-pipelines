"""
Request-args schemas for the load pipeline.

Split the same way the climate pipeline splits its own -- location apart from fetch
config, so neither is duplicated:

1. :class:`LoadGeography` -- one **location**: a single PUMA. Its NHGIS GISJOIN
   embeds the state FIPS, so the state and its standard-time offset are derived
   rather than supplied. There is no *when*: each source publishes one modelled
   year.
2. :class:`LoadGeographies` -- the user-facing location: the **one or more** PUMAs a
   run is asked for. A run-level grouping only; it expands into the per-PUMA
   geographies above, which are what key every write.
3. :class:`ResstockRequestArgs` -- ResStock **fetch config** (which release, which
   upgrade). Not how many buildings: a location means every building in it.
4. :class:`ComstockRequestArgs` -- ComStock **fetch config**, the same shape.
5. :class:`DsgridRequestArgs` -- dsgrid **fetch config** (which of the two ``.dsg``
   source files).
6. :class:`IndustrialLoadRequestArgs` -- the **silver** key: a state plus the dsgrid
   sources combined into that table. Keyed by state, not PUMA, because dsgrid
   publishes per state and covers every county at once.

``silver.py`` combines a geography with one source's fetch config to build each
bronze manifest key.

Beware the name overlap: :class:`DsgridRequestArgs` here is the fetch config, while
``dsgrid.bronze.DsgridRequestArgs`` is that source's *manifest key* (fetch config +
state). Both are always module-qualified at the call site.

The time-axis facts and state offsets below are the pipeline's single source of
truth; ``silver.py`` reads them rather than hardcoding per-source assumptions.
"""

from __future__ import annotations

import datetime as dt
from typing import Self

import pydantic

import common.exceptions
import common.models
from external_data.load_pipeline import oedi_building_stock as oedi
from external_data.load_pipeline.comstock import bronze as comstock_bronze
from external_data.load_pipeline.dsgrid import bronze as dsgrid_bronze
from external_data.load_pipeline.resstock import bronze as resstock_bronze

# --------------------------------------------------------------------------- #
# Geography patterns
# --------------------------------------------------------------------------- #

STATE_PATTERN = oedi.STATE_PATTERN
PUMA_PATTERN = oedi.PUMA_PATTERN

# Both GISJOIN flavors carry the 2-digit state FIPS in the same slice, so a PUMA
# and a county can be checked for the same state without a FIPS lookup table.
_STATE_FIPS_SLICE = slice(1, 3)

# --------------------------------------------------------------------------- #
# dsgrid time axis
# --------------------------------------------------------------------------- #

# dsgrid EFS: hourly 2012 (leap, 8784 h), interval-ending, stated on ONE national
# clock for every county -- not the county's local time. Silver conforms bronze
# rather than re-basing it, so these are published as the facts a consumer needs to
# interpret a timestamp, not as inputs to a conversion here.
DSGRID_INTERVAL_MINUTES = 60
DSGRID_YEAR = 2012
DSGRID_UTC_OFFSET_MINUTES = -300  # EST, no DST

DSGRID_INTERVAL = dt.timedelta(minutes=DSGRID_INTERVAL_MINUTES)

# --------------------------------------------------------------------------- #
# Curated source variables
# --------------------------------------------------------------------------- #

# OEDI publishes ~150 metadata columns and dozens of end-use channels per source.
# We pull only the slice the load work needs: enough geography to place a building,
# enough characteristics to explain its shape, the expansion weight, and electricity
# split into total / cooling / heating -- the end uses that drive weather sensitivity
# and electrification headroom.
#
# Each tuple is derived from that source's raw -> bronze projection map, so the
# variable list and the bronze schema cannot drift apart.

RESSTOCK_METADATA_VARIABLES: tuple[str, ...] = tuple(
    resstock_bronze.METADATA_RAW_TO_COLUMN
)
RESSTOCK_TIMESERIES_VARIABLES: tuple[str, ...] = tuple(
    resstock_bronze.TIMESERIES_RAW_TO_COLUMN
)
COMSTOCK_METADATA_VARIABLES: tuple[str, ...] = tuple(
    comstock_bronze.PUMA_METADATA_RAW_TO_COLUMN
)
COMSTOCK_TIMESERIES_VARIABLES: tuple[str, ...] = tuple(
    comstock_bronze.TIMESERIES_RAW_TO_COLUMN
)

# dsgrid's end uses are not a curated choice: the ``.dsg`` enumeration is fixed and
# validated against the file at reconstruction time, so there is nothing to select.
# Exposed for visibility only -- see DsgridRequestArgs.
DSGRID_INDUSTRIAL_END_USES: tuple[str, ...] = tuple(
    dsgrid_bronze.PROFILES[dsgrid_bronze.DsgridSource.INDUSTRIAL].enduse_ids
)
DSGRID_GAPS_END_USES: tuple[str, ...] = tuple(
    dsgrid_bronze.PROFILES[dsgrid_bronze.DsgridSource.GAPS].enduse_ids
)


def _check_curated(
    value: tuple[str, ...], curated: tuple[str, ...], label: str
) -> tuple[str, ...]:
    """
    Require *value* to be exactly the curated set.

    The climate pipeline lets a request *add* variables, because an extra CDS or
    NSRDB attribute is simply fetched and ignored. Here the projection is driven by
    the bronze raw -> column map, so an added name would be silently dropped instead.
    Widening the set means a schema field and a map entry too -- which changes the
    curated tuple, so anything else is refused rather than quietly ignored.
    """
    missing = [v for v in curated if v not in value]
    extra = [v for v in value if v not in curated]
    if missing or extra:
        msg = (
            f"{label} must be exactly the curated set; "
            f"missing {missing}, unexpected {extra}. Add a bronze schema field "
            "and a raw -> column mapping first if you need another variable."
        )
        raise ValueError(msg)
    # Membership, not order: the projection is driven by the bronze map, so a
    # reordered tuple names the same columns (and comparing tuples would reject it
    # with an empty "missing [], unexpected []"). Returning the curated order also
    # keeps the manifest key stable.
    return curated


# --------------------------------------------------------------------------- #
# State standard-time offsets
# --------------------------------------------------------------------------- #

# Standard-time (winter) offset from UTC in minutes, per state. Both axes we combine
# are standard time year-round -- neither observes DST -- so these are the only
# offsets needed, with no spring-forward gaps or fall-back duplicates to reconcile.
STATE_STANDARD_UTC_OFFSET_MINUTES: dict[str, int] = {
    "AL": -360, "AK": -540, "AZ": -420, "AR": -360, "CA": -480,
    "CO": -420, "CT": -300, "DE": -300, "DC": -300, "FL": -300,
    "GA": -300, "HI": -600, "ID": -420, "IL": -360, "IN": -300,
    "IA": -360, "KS": -360, "KY": -300, "LA": -360, "ME": -300,
    "MD": -300, "MA": -300, "MI": -300, "MN": -360, "MS": -360,
    "MO": -360, "MT": -420, "NE": -360, "NV": -480, "NH": -300,
    "NJ": -300, "NM": -420, "NY": -300, "NC": -300, "ND": -360,
    "OH": -300, "OK": -360, "OR": -480, "PA": -300, "RI": -300,
    "SC": -300, "SD": -360, "TN": -360, "TX": -360, "UT": -420,
    "VT": -300, "VA": -300, "WA": -480, "WV": -300, "WI": -360,
    "WY": -420,
}  # fmt: skip

# States split across two time zones, where the table above holds the *dominant*
# zone only. A PUMA in the minority zone is off by an hour, so a caller who knows
# better should pass ``local_standard_utc_offset_minutes`` explicitly.
MULTI_ZONE_STATES = frozenset(
    {"AK", "FL", "ID", "IN", "KS", "KY", "MI", "NE", "ND", "OR", "SD", "TN", "TX"}
)


# State FIPS -> two-letter code. A PUMA GISJOIN embeds the state FIPS, so this is
# what lets one PUMA id stand in for the whole geography: no separate state field.
STATE_FIPS_TO_CODE: dict[str, str] = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA",
    "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
    "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
    "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
    "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
    "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
    "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
    "56": "WY",
}  # fmt: skip


def state_code_for_fips(fips: str) -> str:
    """
    Two-letter state code for a 2-digit state FIPS.

    Raises:
        PipelineValueError: If *fips* is not a state FIPS (e.g. a territory).
    """
    if fips not in STATE_FIPS_TO_CODE:
        msg = (
            f"unknown state FIPS {fips!r}; the load pipeline covers the 50 states "
            "plus DC, which is what the three sources publish"
        )
        raise common.exceptions.PipelineValueError(msg)
    return STATE_FIPS_TO_CODE[fips]


def standard_utc_offset_minutes(state: str) -> int:
    """
    Standard-time UTC offset for *state*, in minutes.

    Raises:
        PipelineValueError: If *state* is not in
            :data:`STATE_STANDARD_UTC_OFFSET_MINUTES`.
    """
    if state not in STATE_STANDARD_UTC_OFFSET_MINUTES:
        msg = (
            f"no standard-time offset known for state {state!r}; pass "
            "local_standard_utc_offset_minutes on the request"
        )
        raise common.exceptions.PipelineValueError(msg)
    return STATE_STANDARD_UTC_OFFSET_MINUTES[state]


# --------------------------------------------------------------------------- #
# 1. Geography (the "location", shared by every bronze ingest and the silver join)
# --------------------------------------------------------------------------- #


class LoadGeography(common.models.FrozenModel, extra="forbid"):
    """
    Where to build a load profile: one PUMA
    """

    # One field is enough because a PUMA is the finest geography both building-stock
    # sources publish per-building profiles for, and its GISJOIN embeds the state
    # FIPS -- so the state, and with it every file path, is derived:
    #
    # - ResStock / ComStock timeseries are keyed by state + PUMA: a direct match.
    # - ResStock metadata is per state, ComStock's per PUMA; both are addressed from
    #   the same field.
    # - dsgrid publishes per state and covers every county, so it needs only the
    #   state -- one dsgrid bronze serves every PUMA in it.
    #
    # A geography id rather than a lat/long, deliberately: all three sources are
    # census-geography-keyed and none understands a coordinate, so a coordinate door
    # would unlock no data and would add a resolution step plus a boundary-geometry
    # ingest in front of a working key. (The climate pipeline takes coordinates
    # because ERA5 and NSRDB are gridded; each pipeline takes the key its sources
    # have.) A caller holding a coordinate resolves it to a PUMA once, elsewhere --
    # never by centroid or county, which spends a whole ingest on the wrong buildings
    # and returns a plausible number.

    puma_gisjoin: str = pydantic.Field(
        pattern=PUMA_PATTERN,
        title="PUMA GISJOIN",
        description="NHGIS PUMA GISJOIN: 'G' + 2-digit state FIPS + a padding zero "
        "+ the 5-digit PUMA code. Codes are 2010-census vintage.",
    )
    local_standard_utc_offset_minutes: int | None = pydantic.Field(
        default=None,
        ge=-12 * 60,
        le=14 * 60,
        description="the PUMA's standard-time UTC offset. Defaults to the state's "
        "dominant zone; set it explicitly for a PUMA in the minority zone of a "
        "state in MULTI_ZONE_STATES.",
    )

    @property
    def state_fips(self) -> str:
        """The 2-digit state FIPS embedded in the PUMA GISJOIN."""
        return self.puma_gisjoin[_STATE_FIPS_SLICE]

    @property
    def state(self) -> str:
        """
        Two-letter state code, derived from the PUMA -- so it is never supplied
        separately, and a state / PUMA mismatch cannot be expressed.
        """
        return state_code_for_fips(self.state_fips)

    @property
    def utc_offset_minutes(self) -> int:
        """
        The PUMA's standard-time UTC offset: the explicit override when given,
        else the state's dominant zone.
        """
        if self.local_standard_utc_offset_minutes is not None:
            return self.local_standard_utc_offset_minutes
        return standard_utc_offset_minutes(self.state)


# --------------------------------------------------------------------------- #
# 1b. The PUMAs one run is asked for
# --------------------------------------------------------------------------- #

# How many PUMAs a single run may carry. Not a cap on *data* -- a PUMA asked for is
# always ingested whole -- but on how much work one flow run may hold, because a
# flow's timeout is fixed when the module is imported and cannot be widened per run.
# The flows derive their budgets from this number, which is what keeps a parent's
# timeout at or above the work it schedules.
#
# Splitting a larger job across runs costs nothing: the ledger is consulted before
# anything is fetched, so a run overlapping an earlier one pays only for what it
# adds.
MAX_PUMAS_PER_RUN = 20


class LoadGeographies(common.models.FrozenModel, extra="forbid"):
    """
    Where to build load profiles: one or more PUMAs
    """

    # The plural is *run-level* only. Every bronze write stays keyed by a single
    # PUMA (:class:`LoadGeography`), the grain the sources publish at and reuse works
    # at: a list in a manifest key would make "the PUMAs I asked for together" part
    # of a dataset's identity, so the same PUMA in a different group would miss its
    # own data. Asking for a PUMA alone or in a group of five resolves to one key.
    #
    # The PUMAs need not share a state. dsgrid publishes per state and its silver is
    # keyed by state, so the flows collapse the list to its distinct states there.

    pumas: tuple[LoadGeography, ...] = pydantic.Field(
        min_length=1,
        max_length=MAX_PUMAS_PER_RUN,
        title="PUMA GISJOINs",
    )

    @pydantic.model_validator(mode="before")
    @classmethod
    def _accept_a_bare_list(cls, value: object) -> object:
        """A list of codes (or one code) stands for the whole model."""
        if isinstance(value, str | list | tuple):
            return {"pumas": value}
        return value

    @pydantic.field_validator("pumas", mode="before")
    @classmethod
    def _accept_bare_codes(cls, value: object) -> object:
        """
        Let each entry be a GISJOIN string rather than a ``{"puma_gisjoin": ...}``.

        The offset override is rare -- a PUMA in the minority zone of a
        MULTI_ZONE_STATES state -- so requiring an object per PUMA would make every
        run form pay for it. An entry that needs it is still spelled out in full.
        """
        if isinstance(value, str):
            value = [value]
        if isinstance(value, list | tuple):
            return [{"puma_gisjoin": v} if isinstance(v, str) else v for v in value]
        return value

    @pydantic.model_validator(mode="after")
    def _check_unique(self) -> Self:
        """
        Refuse a repeated PUMA rather than quietly dropping it.

        Ingesting it twice is harmless -- the second pass reuses the first's writes --
        which is why silence is wrong: a repeat in a typed-in list is a typo, and a
        run meant to cover five PUMAs would cover four without saying so.
        """
        seen = [geography.puma_gisjoin for geography in self.pumas]
        duplicates = sorted({code for code in seen if seen.count(code) > 1})
        if duplicates:
            msg = f"puma_gisjoins must be unique, got {duplicates} more than once"
            raise common.exceptions.PipelineValueError(msg)
        return self

    @classmethod
    def of(cls, *puma_gisjoins: str) -> Self:
        """
        Build from bare GISJOIN codes -- the typed way to say it in code.

        The validators above accept the same shorthand from a run form's JSON,
        where nothing is typed; this is its counterpart for a caller who is.
        """
        return cls(
            pumas=tuple(LoadGeography(puma_gisjoin=code) for code in puma_gisjoins)
        )

    @property
    def puma_gisjoins(self) -> tuple[str, ...]:
        """The requested PUMA codes, in the order given."""
        return tuple(geography.puma_gisjoin for geography in self.pumas)

    @property
    def states(self) -> tuple[str, ...]:
        """The distinct states covered, ordered by their first PUMA."""
        return tuple(dict.fromkeys(geography.state for geography in self.pumas))

    def by_state(self) -> dict[str, tuple[LoadGeography, ...]]:
        """
        The PUMAs of each distinct state, grouped, in the order they were given.

        What the state-published *fetches* iterate: ResStock's metadata is one file
        per state, so a run reads it once and cuts a write per PUMA -- this grouping
        is what lets the download be hoisted above the keys.
        """
        grouped: dict[str, list[LoadGeography]] = {}
        for geography in self.pumas:
            grouped.setdefault(geography.state, []).append(geography)
        return {state: tuple(pumas) for state, pumas in grouped.items()}

    def one_per_state(self) -> tuple[LoadGeography, ...]:
        """
        The first PUMA given for each distinct state.

        What the state-grained steps iterate: dsgrid covers a whole state at once, so
        running it per PUMA would re-ask the ledger the same question.
        """
        first: dict[str, LoadGeography] = {}
        for geography in self.pumas:
            first.setdefault(geography.state, geography)
        return tuple(first.values())


# --------------------------------------------------------------------------- #
# 2. ResStock request (fetch config)
# --------------------------------------------------------------------------- #


class ResstockRequestArgs(common.models.FrozenModel, extra="forbid"):
    """
    ResStock fetch: which release, and which upgrade of it
    """

    # No geography -- that comes from LoadGeography. ``upgrade`` lives here rather
    # than on the combined request because it is a building-stock concept (a measure
    # package applied to the modelled stock) that dsgrid has no equivalent of.

    source_dataset: str = pydantic.Field(
        default=resstock_bronze.DATASET,
        description="OEDI ResStock release directory to read",
    )
    release_year: int = pydantic.Field(
        default=oedi.RELEASE_YEAR,
        description="OEDI publication year directory (e.g. 2025)",
    )
    upgrade: int = pydantic.Field(
        default=oedi.DEFAULT_UPGRADE,
        ge=0,
        description="upgrade / measure-package id; 0 is the un-retrofit baseline",
    )
    metadata_variables: tuple[str, ...] = pydantic.Field(
        default=RESSTOCK_METADATA_VARIABLES,
        min_length=1,
        description="raw metadata columns projected from the source parquet: "
        "geography, building characteristics, the expansion weight, and annual "
        "electricity totals. A curated slice of the ~150 columns OEDI publishes.",
    )
    timeseries_variables: tuple[str, ...] = pydantic.Field(
        default=RESSTOCK_TIMESERIES_VARIABLES,
        min_length=1,
        description="raw timeseries columns projected from each per-building "
        "parquet: the timestamp, the building id, and whole-building electricity "
        "split into total / cooling / heating.",
    )

    @pydantic.field_validator("metadata_variables")
    @classmethod
    def _check_metadata_variables(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _check_curated(value, RESSTOCK_METADATA_VARIABLES, "metadata_variables")

    @pydantic.field_validator("timeseries_variables")
    @classmethod
    def _check_timeseries_variables(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _check_curated(
            value, RESSTOCK_TIMESERIES_VARIABLES, "timeseries_variables"
        )


# --------------------------------------------------------------------------- #
# 3. ComStock request (fetch config)
# --------------------------------------------------------------------------- #


class ComstockRequestArgs(common.models.FrozenModel, extra="forbid"):
    """
    ComStock fetch: which release, and which upgrade of it
    """

    # Same shape as ResstockRequestArgs, kept separate because the two sources
    # version independently: their releases are different directories on different
    # cadences, and one shared model would force them to move together.

    source_dataset: str = pydantic.Field(
        default=comstock_bronze.DATASET,
        description="OEDI ComStock release directory to read",
    )
    release_year: int = pydantic.Field(
        default=oedi.RELEASE_YEAR,
        description="OEDI publication year directory (e.g. 2025)",
    )
    upgrade: int = pydantic.Field(
        default=oedi.DEFAULT_UPGRADE,
        ge=0,
        description="upgrade / measure-package id; 0 is the un-retrofit baseline",
    )
    metadata_variables: tuple[str, ...] = pydantic.Field(
        default=COMSTOCK_METADATA_VARIABLES,
        min_length=1,
        description="raw metadata columns projected from the source parquet: "
        "geography, building characteristics, the expansion weight, and annual "
        "electricity totals. A curated slice of the ~150 columns OEDI publishes.",
    )
    timeseries_variables: tuple[str, ...] = pydantic.Field(
        default=COMSTOCK_TIMESERIES_VARIABLES,
        min_length=1,
        description="raw timeseries columns projected from each per-building "
        "parquet: the timestamp, the building id, and whole-building electricity "
        "split into total / cooling / heating.",
    )

    @pydantic.field_validator("metadata_variables")
    @classmethod
    def _check_metadata_variables(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _check_curated(value, COMSTOCK_METADATA_VARIABLES, "metadata_variables")

    @pydantic.field_validator("timeseries_variables")
    @classmethod
    def _check_timeseries_variables(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _check_curated(
            value, COMSTOCK_TIMESERIES_VARIABLES, "timeseries_variables"
        )


# --------------------------------------------------------------------------- #
# 4. dsgrid request (fetch config)
# --------------------------------------------------------------------------- #


class DsgridRequestArgs(common.models.FrozenModel, extra="forbid"):
    """
    dsgrid fetch: which of the two industrial ``.dsg`` source files
    """

    # No upgrade or release-year axis (one 2018 EFS submission holds both files) and
    # no variable field: the .dsg end-use enumeration is fixed and validated against
    # the file during reconstruction. See DSGRID_INDUSTRIAL_END_USES /
    # DSGRID_GAPS_END_USES.

    sources: tuple[dsgrid_bronze.DsgridSource, ...] = pydantic.Field(
        default=tuple(dsgrid_bronze.DsgridSource),
        min_length=1,
        description="which dsgrid industrial source file(s) to ingest; both by "
        "default, since together they make up the industrial total",
    )
    source_dataset: str = pydantic.Field(
        default=dsgrid_bronze.SOURCE_DATASET,
        description="OEDI dsgrid EFS submission recorded for provenance",
    )

    @pydantic.model_validator(mode="after")
    def _check_unique(self) -> Self:
        if len(set(self.sources)) != len(self.sources):
            msg = f"sources must be unique, got {list(self.sources)}"
            raise common.exceptions.PipelineValueError(msg)
        # Canonicalise the order: ``model_dump_json`` preserves tuple order and the
        # manifest matches that JSON exactly, so the same files listed differently
        # would key a second, identical dataset -- and a reader spelling it the other
        # way would get a miss and a full rebuild.
        canonical = tuple(s for s in dsgrid_bronze.DsgridSource if s in self.sources)
        if canonical != self.sources:
            object.__setattr__(self, "sources", canonical)
        return self


# --------------------------------------------------------------------------- #
# 5. Industrial silver request
# --------------------------------------------------------------------------- #


class IndustrialLoadRequestArgs(common.models.FrozenModel, extra="forbid"):
    """
    One state's industrial load, stacked from both dsgrid source files
    """

    # Keyed by state, not PUMA: dsgrid publishes per state and covers every county
    # at once, so two PUMAs in one state would key two identical tables. The rest of
    # the pipeline stays PUMA-based because the building-stock sources genuinely are.
    #
    # ``dsgrid`` is part of the key because a table built from the manufacturing file
    # alone is a different quantity from one built from both halves.

    state: str = pydantic.Field(
        default=oedi.DEFAULT_STATE,
        pattern=STATE_PATTERN,
        description="two-letter state code; dsgrid bronze is partitioned under it",
    )
    dsgrid: DsgridRequestArgs = pydantic.Field(
        default_factory=DsgridRequestArgs,
        description="which dsgrid source file(s) the table combines",
    )
