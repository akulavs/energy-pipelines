"""
dsgrid 2018 EFS industrial -- bronze ingestion (both source files)

- ``DsgridSource.INDUSTRIAL`` (``industrial.dsg``) -- manufacturing demand by
  4-digit NAICS *subsector*, broken out into 12 electricity end uses. Writes
  ``dsgrid_industrial_metadata_bronze`` + ``dsgrid_industrial_timeseries_bronze``.
- ``DsgridSource.GAPS`` (``industrial_gaps.dsg``) -- the non-manufacturing
  industrial *sectors* (agriculture, mining, construction) by 2-digit NAICS,
  carrying a single un-decomposed end use. Writes
  ``dsgrid_industrial_gaps_metadata_bronze`` +
  ``dsgrid_industrial_gaps_timeseries_bronze``.
"""

from __future__ import annotations
import dataclasses
from collections.abc import Sequence
import datetime as dt
import logging
from pathlib import Path
import numpy as np
import patito as pt
import polars as pl
import pydantic
import httpx
import common.exceptions
import common.models
from common.frames import BaseDataFrameSchema
from common.storage import columnar
from common.storage.manifest import ManifestRow
from external_data.load_pipeline import dsg_common

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Shared dsgrid constants (bucket, prefix, submission, axis) live in dsg_common;
# re-exported here for the module's public surface.
SOURCE_DATASET = dsg_common.SOURCE_DATASET
N_HOURS = dsg_common.N_HOURS

DEFAULT_BASE_DIR = "bronze"

DEFAULT_STATE = "DC"

# One writer for every dsgrid dataset, both source files.
DEFAULT_WRITER = "dsgrid_bronze"

# Dataset names as module constants (AGENTS.md: never inline a dataset name).
INDUSTRIAL_METADATA_DATASET_NAME = "dsgrid_industrial_metadata_bronze"
INDUSTRIAL_TIMESERIES_DATASET_NAME = "dsgrid_industrial_timeseries_bronze"
GAPS_METADATA_DATASET_NAME = "dsgrid_industrial_gaps_metadata_bronze"
GAPS_TIMESERIES_DATASET_NAME = "dsgrid_industrial_gaps_timeseries_bronze"

# Re-exported for the module's public surface / tests
_fips_to_county_gisjoin = dsg_common.fips_to_county_gisjoin

# The 12 industrial electricity end uses, in enumeration order. Snake-case
# enumeration id -> clean bronze column stem
_INDUSTRIAL_ENDUSE_IDS: tuple[str, ...] = (
    "conventional_boiler_use",
    "process_heating",
    "process_cooling_and_refrigeration",
    "machine_drive",
    "electro_chemical_processes",
    "other_process_use",
    "facility_hvac",
    "facility_lighting",
    "other_facility_support",
    "onsite_transportation",
    "other_nonprocess_use",
    "end_use_not_reported",
)

# The gaps file carries a single, un-decomposed end use
_GAPS_ENDUSE_IDS: tuple[str, ...] = ("energy_consumption",)


# --------------------------------------------------------------------------- #
# Source selector + request
# --------------------------------------------------------------------------- #


class DsgridSource(common.models.CaseInsensitiveStrEnum):
    """Which dsgrid EFS industrial ``.dsg`` file a request targets."""

    INDUSTRIAL = "industrial"
    GAPS = "industrial_gaps"


class DsgridRequestArgs(common.models.FrozenModel, extra="forbid"):
    """
    A whole-state dsgrid industrial demand request

    ``source`` selects which national ``.dsg`` file to reconstruct; ``state``
    selects the slice. Both the metadata and timeseries bronze datasets for a
    source use this same request, so the pair always covers the same geography.
    """

    source: DsgridSource = pydantic.Field(
        default=DsgridSource.INDUSTRIAL,
        description="which dsgrid EFS industrial file to reconstruct "
        "(industrial.dsg vs industrial_gaps.dsg)",
    )
    source_dataset: str = pydantic.Field(
        default=SOURCE_DATASET,
        description="OEDI dsgrid EFS submission recorded for provenance; a single "
        "submission holds both files today, so this does not change object_url()",
    )
    state: str = pydantic.Field(
        default=DEFAULT_STATE,
        pattern=dsg_common.STATE_PATTERN,
        description="two-letter state code to reconstruct all counties for",
    )

    def object_url(self) -> str:
        """
        Full HTTPS URL of the national ``.dsg`` file for ``source`` on the OEDI lake
        """
        dsg_file = PROFILES[self.source].dsg_file
        return f"{dsg_common.BASE_URL}/{dsg_common.OEDI_PREFIX}/{dsg_file}"


# --------------------------------------------------------------------------- #
# Dataset schemas (catalog / manifest)
# --------------------------------------------------------------------------- #


class DsgridIndustrialMetadataBronzeSchema(BaseDataFrameSchema):
    """
    One row per county x NAICS subsector: geography + annual end-use totals
    """

    state: str = pt.Field(dtype=pl.String, description="two-letter state code")
    county_gisjoin: str = pt.Field(
        dtype=pl.String,
        description="NHGIS county GISJOIN (derived from the 5-digit FIPS)",
    )
    county_fips: str = pt.Field(
        dtype=pl.String, description="5-digit county FIPS code (source geography id)"
    )
    county_name: str = pt.Field(
        dtype=pl.String, description="county name (e.g. 'Autauga County, AL')"
    )
    naics_subsector: str = pt.Field(
        dtype=pl.String, description="4-digit NAICS manufacturing subsector code"
    )
    subsector_name: str = pt.Field(dtype=pl.String, description="NAICS subsector name")
    annual_electricity_total_mwh: float = pt.Field(
        dtype=pl.Float64,
        description="annual total industrial electricity (MWh); sum of the 12 end uses",
    )
    annual_electricity_conventional_boiler_use_mwh: float = pt.Field(
        dtype=pl.Float64, description="annual conventional boiler electricity (MWh)"
    )
    annual_electricity_process_heating_mwh: float = pt.Field(
        dtype=pl.Float64, description="annual process-heating electricity (MWh)"
    )
    annual_electricity_process_cooling_and_refrigeration_mwh: float = pt.Field(
        dtype=pl.Float64,
        description="annual process cooling & refrigeration electricity (MWh)",
    )
    annual_electricity_machine_drive_mwh: float = pt.Field(
        dtype=pl.Float64, description="annual machine-drive electricity (MWh)"
    )
    annual_electricity_electro_chemical_processes_mwh: float = pt.Field(
        dtype=pl.Float64,
        description="annual electro-chemical process electricity (MWh)",
    )
    annual_electricity_other_process_use_mwh: float = pt.Field(
        dtype=pl.Float64, description="annual other process-use electricity (MWh)"
    )
    annual_electricity_facility_hvac_mwh: float = pt.Field(
        dtype=pl.Float64, description="annual facility HVAC electricity (MWh)"
    )
    annual_electricity_facility_lighting_mwh: float = pt.Field(
        dtype=pl.Float64, description="annual facility lighting electricity (MWh)"
    )
    annual_electricity_other_facility_support_mwh: float = pt.Field(
        dtype=pl.Float64, description="annual other facility-support electricity (MWh)"
    )
    annual_electricity_onsite_transportation_mwh: float = pt.Field(
        dtype=pl.Float64, description="annual on-site transportation electricity (MWh)"
    )
    annual_electricity_other_nonprocess_use_mwh: float = pt.Field(
        dtype=pl.Float64, description="annual other non-process electricity (MWh)"
    )
    annual_electricity_end_use_not_reported_mwh: float = pt.Field(
        dtype=pl.Float64,
        description="annual electricity with end use not reported (MWh)",
    )


class DsgridIndustrialTimeseriesBronzeSchema(BaseDataFrameSchema):
    """
    One row per county x NAICS subsector x hour: the industrial load profile
    """

    timestamp: dt.datetime = pt.Field(
        dtype=pl.Datetime(time_unit="us"),
        description="interval-ending timestamp (local standard time, 2012 EST; naive)",
    )
    county_gisjoin: str = pt.Field(
        dtype=pl.String, description="NHGIS county GISJOIN the segment belongs to"
    )
    state: str = pt.Field(dtype=pl.String, description="two-letter state code")
    naics_subsector: str = pt.Field(
        dtype=pl.String, description="4-digit NAICS manufacturing subsector code"
    )
    electricity_total_mwh: float = pt.Field(
        dtype=pl.Float32,
        description="total industrial electricity for the interval (MWh); "
        "sum of the 12 end uses",
    )
    electricity_conventional_boiler_use_mwh: float = pt.Field(
        dtype=pl.Float32, description="conventional boiler electricity (MWh)"
    )
    electricity_process_heating_mwh: float = pt.Field(
        dtype=pl.Float32, description="process-heating electricity (MWh)"
    )
    electricity_process_cooling_and_refrigeration_mwh: float = pt.Field(
        dtype=pl.Float32,
        description="process cooling & refrigeration electricity (MWh)",
    )
    electricity_machine_drive_mwh: float = pt.Field(
        dtype=pl.Float32, description="machine-drive electricity (MWh)"
    )
    electricity_electro_chemical_processes_mwh: float = pt.Field(
        dtype=pl.Float32, description="electro-chemical process electricity (MWh)"
    )
    electricity_other_process_use_mwh: float = pt.Field(
        dtype=pl.Float32, description="other process-use electricity (MWh)"
    )
    electricity_facility_hvac_mwh: float = pt.Field(
        dtype=pl.Float32, description="facility HVAC electricity (MWh)"
    )
    electricity_facility_lighting_mwh: float = pt.Field(
        dtype=pl.Float32, description="facility lighting electricity (MWh)"
    )
    electricity_other_facility_support_mwh: float = pt.Field(
        dtype=pl.Float32, description="other facility-support electricity (MWh)"
    )
    electricity_onsite_transportation_mwh: float = pt.Field(
        dtype=pl.Float32, description="on-site transportation electricity (MWh)"
    )
    electricity_other_nonprocess_use_mwh: float = pt.Field(
        dtype=pl.Float32, description="other non-process electricity (MWh)"
    )
    electricity_end_use_not_reported_mwh: float = pt.Field(
        dtype=pl.Float32, description="electricity with end use not reported (MWh)"
    )


class DsgridIndustrialGapsMetadataBronzeSchema(BaseDataFrameSchema):
    """
    One row per county x NAICS sector: geography + the annual electricity total
    """

    state: str = pt.Field(dtype=pl.String, description="two-letter state code")
    county_gisjoin: str = pt.Field(
        dtype=pl.String,
        description="NHGIS county GISJOIN (derived from the 5-digit FIPS)",
    )
    county_fips: str = pt.Field(
        dtype=pl.String, description="5-digit county FIPS code (source geography id)"
    )
    county_name: str = pt.Field(
        dtype=pl.String, description="county name (e.g. 'Autauga County, AL')"
    )
    naics_sector: str = pt.Field(
        dtype=pl.String,
        description="2-digit NAICS non-manufacturing industrial sector code "
        "(11 agriculture, 21 mining, 23 construction)",
    )
    sector_name: str = pt.Field(dtype=pl.String, description="NAICS sector name")
    annual_electricity_total_mwh: float = pt.Field(
        dtype=pl.Float64, description="annual total electricity for the sector (MWh)"
    )


class DsgridIndustrialGapsTimeseriesBronzeSchema(BaseDataFrameSchema):
    """
    One row per county x NAICS sector x hour: the electricity load profile
    """

    timestamp: dt.datetime = pt.Field(
        dtype=pl.Datetime(time_unit="us"),
        description="interval-ending timestamp (local standard time, 2012 EST; naive)",
    )
    county_gisjoin: str = pt.Field(
        dtype=pl.String, description="NHGIS county GISJOIN the segment belongs to"
    )
    state: str = pt.Field(dtype=pl.String, description="two-letter state code")
    naics_sector: str = pt.Field(
        dtype=pl.String, description="2-digit NAICS non-manufacturing sector code"
    )
    electricity_total_mwh: float = pt.Field(
        dtype=pl.Float32,
        description="total electricity for the sector for the interval (MWh)",
    )


# --------------------------------------------------------------------------- #
# Source profiles (what differs between the two .dsg files)
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class DsgridProfile:
    """
    What differs between the two source files -- primitives only

    The column layout (which columns, which keys) is *derived* from these via
    properties, so it cannot drift. ``decompose_enduses`` is the one behavioural
    switch: industrial breaks each end use into its own column, gaps folds its single
    end use into the total only.
    """

    source: DsgridSource
    enduse_ids: tuple[str, ...]  # the file's end-use enumeration (fail-loud check)
    decompose_enduses: bool
    sector_column: str  # naics_subsector / naics_sector
    sector_name_column: str  # subsector_name / sector_name
    metadata_dataset_name: str
    timeseries_dataset_name: str
    metadata_schema: type[BaseDataFrameSchema]
    timeseries_schema: type[BaseDataFrameSchema]

    @property
    def dsg_file(self) -> str:
        # The enum value is the file stem, e.g. "industrial_gaps" -> the .dsg.
        return f"{self.source.value}.dsg"

    @property
    def sector_noun(self) -> str:
        # "naics_subsector" -> "subsector"; "naics_sector" -> "sector".
        return self.sector_column.removeprefix("naics_")

    @property
    def _column_enduse_ids(self) -> tuple[str, ...]:
        return self.enduse_ids if self.decompose_enduses else ()

    @property
    def annual_enduse_columns(self) -> tuple[str, ...]:
        return tuple(f"annual_electricity_{e}_mwh" for e in self._column_enduse_ids)

    @property
    def timeseries_enduse_columns(self) -> tuple[str, ...]:
        return tuple(f"electricity_{e}_mwh" for e in self._column_enduse_ids)

    @property
    def metadata_columns(self) -> tuple[str, ...]:
        return (
            "state",
            "county_gisjoin",
            "county_fips",
            "county_name",
            self.sector_column,
            self.sector_name_column,
            "annual_electricity_total_mwh",
            *self.annual_enduse_columns,
        )

    @property
    def timeseries_columns(self) -> tuple[str, ...]:
        return (
            "timestamp",
            "county_gisjoin",
            "state",
            self.sector_column,
            "electricity_total_mwh",
            *self.timeseries_enduse_columns,
        )

    @property
    def metadata_key_columns(self) -> tuple[str, ...]:
        return ("county_gisjoin", self.sector_column)

    @property
    def timeseries_key_columns(self) -> tuple[str, ...]:
        return ("timestamp", "county_gisjoin", self.sector_column)

    # No partition columns, for either table. A write is one state, so a
    # ``state=XX/`` directory would hold one file and repeat the key; splitting the
    # timeseries by county instead scattered a state across 56-62 files (~75 KB each
    # for gaps) for a pruning nothing performs -- silver and every other reader take
    # the whole table. The manifest is the index.


# Keyed by ``profile.source`` so the key can't drift from the profile.
PROFILES: dict[DsgridSource, DsgridProfile] = {
    profile.source: profile
    for profile in (
        DsgridProfile(
            source=DsgridSource.INDUSTRIAL,
            enduse_ids=_INDUSTRIAL_ENDUSE_IDS,
            decompose_enduses=True,
            sector_column="naics_subsector",
            sector_name_column="subsector_name",
            metadata_dataset_name=INDUSTRIAL_METADATA_DATASET_NAME,
            timeseries_dataset_name=INDUSTRIAL_TIMESERIES_DATASET_NAME,
            metadata_schema=DsgridIndustrialMetadataBronzeSchema,
            timeseries_schema=DsgridIndustrialTimeseriesBronzeSchema,
        ),
        DsgridProfile(
            source=DsgridSource.GAPS,
            enduse_ids=_GAPS_ENDUSE_IDS,
            decompose_enduses=False,
            sector_column="naics_sector",
            sector_name_column="sector_name",
            metadata_dataset_name=GAPS_METADATA_DATASET_NAME,
            timeseries_dataset_name=GAPS_TIMESERIES_DATASET_NAME,
            metadata_schema=DsgridIndustrialGapsMetadataBronzeSchema,
            timeseries_schema=DsgridIndustrialGapsTimeseriesBronzeSchema,
        ),
    )
}

# Every source needs a profile, or object_url()/ingest fails with a bare KeyError far
# from here. Checked at import, and raised rather than asserted so -O keeps it.
if PROFILES.keys() != set(DsgridSource):
    raise common.exceptions.PipelineError(
        "every DsgridSource needs a profile in PROFILES; "
        f"missing {set(DsgridSource) - PROFILES.keys()}"
    )


# --------------------------------------------------------------------------- #
# Reconstruct -> annual / hourly frames
# --------------------------------------------------------------------------- #
#
# The HDF5 fetch + per-state reconstruction is shared across dsgrid files and lives
# in ``external_data.load_pipeline.dsg_common``; this module only maps reconstructed
# segments onto its own schemas.


def reconstruct_metadata_table(
    data: bytes, state: str, source: DsgridSource = DsgridSource.INDUSTRIAL
) -> pl.DataFrame:
    """
    Reconstruct the annual-rollup metadata table for *state* from the .dsg bytes
    """
    profile = PROFILES[source]
    enums, segments = dsg_common.reconstruct_state(data, state, profile.enduse_ids)
    return _build_metadata_table(enums, segments, state, profile)


def reconstruct_timeseries_table(
    data: bytes, state: str, source: DsgridSource = DsgridSource.INDUSTRIAL
) -> pl.DataFrame:
    """
    Reconstruct the hourly timeseries table for *state* from the .dsg bytes
    """
    profile = PROFILES[source]
    enums, segments = dsg_common.reconstruct_state(data, state, profile.enduse_ids)
    return _build_timeseries_table(enums, segments, state, profile)


def reconstruct_tables(
    data: bytes, state: str, source: DsgridSource = DsgridSource.INDUSTRIAL
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Reconstruct both bronze tables from a single parse of the .dsg bytes

    Use this when producing both datasets in one run, so the expensive HDF5
    reconstruction runs once rather than once per dataset.
    """
    profile = PROFILES[source]
    enums, segments = dsg_common.reconstruct_state(data, state, profile.enduse_ids)
    return (
        _build_metadata_table(enums, segments, state, profile),
        _build_timeseries_table(enums, segments, state, profile),
    )


def _build_metadata_table(
    enums: dsg_common.Enumerations,
    segments: list[dsg_common.Segment],
    state: str,
    profile: DsgridProfile,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for seg in segments:
        per_enduse_annual = seg.hourly.sum(axis=1)  # per-enduse annual MWh
        fips = enums.geo_fips[seg.county_idx]
        row: dict[str, object] = {
            "state": state,
            "county_gisjoin": _fips_to_county_gisjoin(fips),
            "county_fips": fips,
            "county_name": enums.geo_name[seg.county_idx],
            profile.sector_column: seg.naics,
            profile.sector_name_column: enums.sector_name[seg.naics],
            "annual_electricity_total_mwh": float(per_enduse_annual.sum()),
        }
        # per-end-use breakout columns (none for the gaps single-end-use source)
        for i, col in enumerate(profile.annual_enduse_columns):
            row[col] = float(per_enduse_annual[i])
        rows.append(row)

    frame = (
        pl.DataFrame(rows) if rows else dsg_common.empty_frame(profile.metadata_columns)
    )
    return dsg_common.shape_to_schema(
        frame,
        profile.metadata_schema,
        profile.metadata_columns,
        profile.metadata_key_columns,
    )


def _build_timeseries_table(
    enums: dsg_common.Enumerations,
    segments: list[dsg_common.Segment],
    state: str,
    profile: DsgridProfile,
) -> pl.DataFrame:
    n_hours = len(enums.timestamps)

    if not segments:
        frame = dsg_common.empty_frame(profile.timeseries_columns)
        return dsg_common.shape_to_schema(
            frame,
            profile.timeseries_schema,
            profile.timeseries_columns,
            profile.timeseries_key_columns,
        )

    # Assemble columns with numpy, then a single DataFrame build. Each
    # seg.hourly is (n_enduses, n_hours); transpose + stack to (rows, n_enduses).
    enduse_block = np.concatenate([seg.hourly.T for seg in segments], axis=0)
    total = enduse_block.sum(axis=1)
    timestamps = enums.timestamps * len(segments)
    gisjoin = np.repeat(
        [_fips_to_county_gisjoin(enums.geo_fips[seg.county_idx]) for seg in segments],
        n_hours,
    )
    naics = np.repeat([seg.naics for seg in segments], n_hours)

    columns: dict[str, object] = {
        "timestamp": timestamps,
        "county_gisjoin": gisjoin,
        "state": np.repeat(state, len(total)),
        profile.sector_column: naics,
        "electricity_total_mwh": total,
    }
    # per-end-use breakout columns (none for the gaps single-end-use source)
    for i, col in enumerate(profile.timeseries_enduse_columns):
        columns[col] = enduse_block[:, i]

    frame = pl.DataFrame(columns)
    return dsg_common.shape_to_schema(
        frame,
        profile.timeseries_schema,
        profile.timeseries_columns,
        profile.timeseries_key_columns,
    )


# --------------------------------------------------------------------------- #
# Write -> permanent storage
# --------------------------------------------------------------------------- #


def write_metadata_bronze(
    table: pl.DataFrame,
    params: DsgridRequestArgs,
    root_uri: str | Path,
    writer: str = DEFAULT_WRITER,
    write_time: dt.datetime | None = None,
) -> ManifestRow:
    """
    Write the metadata bronze frame for ``params.source`` through the catalog
    """
    profile = PROFILES[params.source]
    return columnar.write_dataset(
        table,
        profile.metadata_schema,
        profile.metadata_dataset_name,
        params,
        str(root_uri),
        writer=writer,
        write_time=write_time,
    )


def write_timeseries_bronze(
    table: pl.DataFrame,
    params: DsgridRequestArgs,
    root_uri: str | Path,
    writer: str = DEFAULT_WRITER,
    write_time: dt.datetime | None = None,
) -> ManifestRow:
    """
    Write the timeseries bronze frame for ``params.source`` through the catalog
    """
    profile = PROFILES[params.source]
    return columnar.write_dataset(
        table,
        profile.timeseries_schema,
        profile.timeseries_dataset_name,
        params,
        str(root_uri),
        writer=writer,
        write_time=write_time,
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def ingest_metadata_bronze(
    args: DsgridRequestArgs | None = None,
    root_uri: str | Path = DEFAULT_BASE_DIR,
    writer: str = DEFAULT_WRITER,
    client: httpx.Client | None = None,
    write_time: dt.datetime | None = None,
    dsg_bytes: bytes | None = None,
) -> ManifestRow:
    """
    Full metadata bronze step: fetch the .dsg, reconstruct the state, write bronze

    Pass ``dsg_bytes`` to reuse an already-downloaded file (e.g. when building
    both datasets in one run).
    """
    if args is None:
        args = DsgridRequestArgs()
    profile = PROFILES[args.source]
    root_uri = dsg_common.resolve_root_uri(root_uri)

    data = dsg_common.load_dsg_bytes(args.object_url(), dsg_bytes, client)
    table = reconstruct_metadata_table(data, args.state, args.source)
    logger.info(
        "metadata bronze [%s]: state %s, %d segment(s), %d %s(s)",
        args.source,
        args.state,
        table.height,
        table[profile.sector_column].n_unique(),
        profile.sector_noun,
    )

    row = write_metadata_bronze(
        table, params=args, root_uri=root_uri, writer=writer, write_time=write_time
    )
    logger.info(
        "metadata bronze [%s]: wrote %s -> %s",
        args.source,
        profile.metadata_dataset_name,
        row.data_uri,
    )
    return row


def ingest_timeseries_bronze(
    args: DsgridRequestArgs | None = None,
    root_uri: str | Path = DEFAULT_BASE_DIR,
    writer: str = DEFAULT_WRITER,
    client: httpx.Client | None = None,
    write_time: dt.datetime | None = None,
    dsg_bytes: bytes | None = None,
) -> ManifestRow:
    """
    Full timeseries bronze step: fetch the .dsg, reconstruct the state, write bronze
    """
    if args is None:
        args = DsgridRequestArgs()
    profile = PROFILES[args.source]
    root_uri = dsg_common.resolve_root_uri(root_uri)

    data = dsg_common.load_dsg_bytes(args.object_url(), dsg_bytes, client)
    table = reconstruct_timeseries_table(data, args.state, args.source)
    logger.info(
        "timeseries bronze [%s]: state %s, %d row(s), %d segment(s)",
        args.source,
        args.state,
        table.height,
        table.select(profile.timeseries_key_columns[1:]).unique().height,
    )

    row = write_timeseries_bronze(
        table, params=args, root_uri=root_uri, writer=writer, write_time=write_time
    )
    logger.info(
        "timeseries bronze [%s]: wrote %s -> %s",
        args.source,
        profile.timeseries_dataset_name,
        row.data_uri,
    )
    return row


def ingest_all(
    state: str = DEFAULT_STATE,
    root_uri: str | Path = DEFAULT_BASE_DIR,
    writer: str = DEFAULT_WRITER,
    client: httpx.Client | None = None,
    write_time: dt.datetime | None = None,
    sources: Sequence[DsgridSource] | None = None,
    requests: Sequence[DsgridRequestArgs] | None = None,
) -> list[ManifestRow]:
    """
    Ingest the requested dsgrid sources in one call for ``state``

    Per source: one download, one parse, then both bronze datasets written. The
    sources stay separate datasets -- this is a convenience over the per-source
    ingest functions, not a merge. A supplied ``client`` is reused across sources.

    ``sources`` defaults to both, which is what the industrial total needs:
    ``industrial.dsg`` is manufacturing only and ``industrial_gaps.dsg`` the
    non-manufacturing remainder. Narrow it only to want one half deliberately.

    Pass ``requests`` to supply the bronze keys directly, and prefer it from a caller
    that also reads these datasets back: otherwise the key is rebuilt from ``state``
    + ``source`` and takes the *default* for every other field, so a caller that set
    ``source_dataset`` would write under one key and resolve under another.
    ``requests`` takes precedence over ``state`` / ``sources``.

    Returns the manifest rows in write order: (metadata, timeseries) per source.
    """
    root_uri = dsg_common.resolve_root_uri(root_uri)
    if requests is None:
        requests = [
            DsgridRequestArgs(source=source, state=state)
            for source in (sources if sources is not None else PROFILES)
        ]
    rows: list[ManifestRow] = []
    for args in requests:
        source = args.source
        # Reconstruct once per source, write both datasets (avoids a re-parse)
        data = dsg_common.load_dsg_bytes(args.object_url(), None, client)
        meta_table, ts_table = reconstruct_tables(data, args.state, source)
        rows.append(
            write_metadata_bronze(
                meta_table, args, root_uri, writer=writer, write_time=write_time
            )
        )
        rows.append(
            write_timeseries_bronze(
                ts_table, args, root_uri, writer=writer, write_time=write_time
            )
        )
        logger.info("ingest_all [%s]: wrote metadata + timeseries", source)
    return rows


if __name__ == "__main__":
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for _row in ingest_all():
        print(_row.data_uri)
