"""
The domain half of the load data-fetch helper: what already exists, and how to
read it back.

The counterpart of ``data_fetch.climate`` for the load pipeline, and deliberately
not the orchestration -- running the suite means the concurrent fan-out in
``batch_jobs.load_pipeline.flows.run_load_pipeline``, and a package may not import
Prefect.

**Six datasets**, so this describes a write as a :class:`DatasetSlice` --
which dataset, which key, how to read it -- and works over lists of those,
instead of six near-identical code paths. The slices for a request come from
:func:`plan_slices`.

The six are the ones a caller asked for: ResStock and ComStock bronze, metadata
and timeseries apiece, and the two industrial silver tables. The dsgrid bronze
the silver is built from is deliberately not among them -- it is an input to the
join, and a caller wanting industrial load wants the joined table.

**Coverage is key equality, not range containment.** Each source publishes one
modelled year and no request narrows it, so a key either exists or it does not;
``load_pipeline.coverage`` says as much and is what this defers to. That removes
the climate side's whole slicing problem: a copy between stores is the whole
file, and there is no "a wider write covers a narrower ask" case to get right.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import logging
import os
import pathlib
from collections.abc import Callable, Sequence

import polars as pl
import pydantic

from common.storage import columnar
from common.storage.columnar import BaseDataFrameSchema
from external_data.data_fetch import geographies
from external_data.load_pipeline import coverage, schema, silver
from external_data.load_pipeline.comstock import bronze as comstock_bronze
from external_data.load_pipeline.resstock import bronze as resstock_bronze

logger = logging.getLogger(__name__)

# Relative fallback for an ad-hoc call, matching the flows' own default.
DEFAULT_ROOT = "load_pipeline"
DEFAULT_WRITER = "load_data_fetch"

# The store every run reads and writes, named by environment rather than
# hard-coded: the path is per-machine (a mounted drive, a sync folder) and baking
# one in would embed one engineer's home directory in the repo.
ROOT_ENV = "LOAD_ROOT"

# The two silver datasets, named as a group. ``industrial_load_silver`` rebuilds
# both for *every state in the request* -- it has no per-state coverage check --
# which is why a run is handed only the geographies that need work.
SILVER_DATASETS = (silver.METADATA_DATASET_NAME, silver.TIMESERIES_DATASET_NAME)


def default_root() -> str:
    """The store named in the environment, or the relative fallback."""
    return os.environ.get(ROOT_ENV) or DEFAULT_ROOT


def resolve_root(root_uri: str | pathlib.Path) -> str:
    """A relative root has no scheme; resolve it so every reader and the flow
    agree on which store is meant."""
    if "://" not in str(root_uri):
        return str(pathlib.Path(root_uri).resolve())
    return str(root_uri)


@dataclasses.dataclass(frozen=True, kw_only=True)
class DatasetSlice:
    """
    One write's identity: which dataset, which key, and how to read it.

    ``read`` is a per-slice callable rather than a shared ``read_dataset`` call
    because the two building-stock timeseries readers do more than resolve a key
    -- they match on a subset for the sake of writes already on disk, and can
    report a PUMA that came back short. Carrying the reader with the slice keeps
    those special cases where they belong instead of in a branch here.
    """

    dataset_name: str
    params: pydantic.BaseModel
    schema_cls: type[BaseDataFrameSchema]
    label: str
    read: Callable[[str, dt.datetime | None], pl.DataFrame]

    @property
    def key(self) -> str:
        """The manifest key this slice resolves by."""
        return self.params.model_dump_json()


@dataclasses.dataclass(frozen=True, kw_only=True)
class Coverage:
    """Which of a request's slices a store already holds."""

    covered: tuple[DatasetSlice, ...]
    missing: tuple[DatasetSlice, ...]

    @property
    def complete(self) -> bool:
        return not self.missing

    def by_dataset(self) -> dict[str, tuple[int, int]]:
        """``{dataset: (covered, missing)}``, for a one-line summary."""
        counts: dict[str, list[int]] = {}
        for index, group in ((0, self.covered), (1, self.missing)):
            for entry in group:
                counts.setdefault(entry.dataset_name, [0, 0])[index] += 1
        return {name: (c, m) for name, (c, m) in counts.items()}


def describe_coverage(cover: Coverage) -> str:
    """A one-line summary of what a store holds, per dataset."""
    if not cover.covered and not cover.missing:
        return "no slices planned"
    return "; ".join(
        f"{name} {have} covered/{miss} missing"
        for name, (have, miss) in sorted(cover.by_dataset().items())
    )


def plan_slices(
    grid: geographies.ConsolidatedGeographies,
    *,
    resstock: schema.ResstockRequestArgs | None = None,
    comstock: schema.ComstockRequestArgs | None = None,
    dsgrid: schema.DsgridRequestArgs | None = None,
) -> list[DatasetSlice]:
    """
    Every write a request resolves to, across all ten datasets.

    The keys come from ``load_pipeline.silver``'s own derivation helpers rather
    than being rebuilt here, so a slice is keyed exactly as the pipeline writes
    it. A key built independently would drift and silently miss its own data.

    Building-stock slices are per PUMA; dsgrid and the industrial silver are per
    state, since dsgrid publishes per state and covers every county at once.
    """
    resstock = resstock or schema.ResstockRequestArgs()
    comstock = comstock or schema.ComstockRequestArgs()
    dsgrid = dsgrid or schema.DsgridRequestArgs()
    slices: list[DatasetSlice] = []

    for geography in grid.pumas:
        puma = geography.puma_gisjoin
        rs_meta = silver.resstock_metadata_request(geography, resstock)
        rs_ts = silver.resstock_timeseries_request(geography, resstock)
        cs_meta = silver.comstock_puma_metadata_request(geography, comstock)
        cs_ts = silver.comstock_timeseries_request(geography, comstock)
        slices += [
            DatasetSlice(
                dataset_name=resstock_bronze.METADATA_DATASET_NAME,
                params=rs_meta,
                schema_cls=resstock_bronze.ResstockMetadataBronzeSchema,
                label=f"resstock metadata {puma}",
                read=lambda root, as_of, a=rs_meta: (
                    resstock_bronze.read_metadata_bronze(a, root, as_of)
                ),
            ),
            DatasetSlice(
                dataset_name=resstock_bronze.TIMESERIES_DATASET_NAME,
                params=rs_ts,
                schema_cls=resstock_bronze.ResstockTimeseriesBronzeSchema,
                label=f"resstock timeseries {puma}",
                read=lambda root, as_of, a=rs_ts: (
                    resstock_bronze.read_puma_timeseries_bronze(a, root, as_of)
                ),
            ),
            DatasetSlice(
                dataset_name=comstock_bronze.PUMA_METADATA_DATASET_NAME,
                params=cs_meta,
                schema_cls=comstock_bronze.ComstockPumaMetadataBronzeSchema,
                label=f"comstock metadata {puma}",
                read=lambda root, as_of, a=cs_meta: (
                    comstock_bronze.read_puma_metadata_bronze(a, root, as_of)
                ),
            ),
            DatasetSlice(
                dataset_name=comstock_bronze.TIMESERIES_DATASET_NAME,
                params=cs_ts,
                schema_cls=comstock_bronze.ComstockTimeseriesBronzeSchema,
                label=f"comstock timeseries {puma}",
                read=lambda root, as_of, a=cs_ts: (
                    comstock_bronze.read_puma_timeseries_bronze(a, root, as_of)
                ),
            ),
        ]

    for state in grid.states:
        # The dsgrid bronze is not planned here. It is an *input* to the silver
        # join, not something a caller asked for -- and the silver is keyed by
        # state, so a state whose silver exists needs no dsgrid bronze read at
        # all. When the silver is missing the pipeline runs, and the flow fetches
        # whatever dsgrid it needs behind its own coverage check.
        industrial = schema.IndustrialLoadRequestArgs(state=state, dsgrid=dsgrid)
        slices += [
            DatasetSlice(
                dataset_name=silver.METADATA_DATASET_NAME,
                params=industrial,
                schema_cls=silver.DsgridIndustrialMetadataSilverSchema,
                label=f"industrial metadata silver {state}",
                read=lambda root, as_of, a=industrial: silver.read_metadata_silver(
                    a, root, as_of
                ),
            ),
            DatasetSlice(
                dataset_name=silver.TIMESERIES_DATASET_NAME,
                params=industrial,
                schema_cls=silver.DsgridIndustrialTimeseriesSilverSchema,
                label=f"industrial timeseries silver {state}",
                read=lambda root, as_of, a=industrial: silver.read_timeseries_silver(
                    a, root, as_of
                ),
            ),
        ]

    return slices


def _dataset_reader(
    dataset_name: str, params: pydantic.BaseModel, schema_cls: type[BaseDataFrameSchema]
) -> Callable[[str, dt.datetime | None], pl.DataFrame]:
    """The plain manifest-resolving read, for datasets with no special reader."""

    def _read(root: str, as_of: dt.datetime | None) -> pl.DataFrame:
        return columnar.read_dataset(
            schema_cls, dataset_name, params, root, as_of=as_of
        )

    return _read


def plan_coverage(
    slices: Sequence[DatasetSlice],
    root_uri: str | pathlib.Path,
    *,
    as_of: dt.datetime | None = None,
) -> Coverage:
    """
    Which of *slices* a store already holds.

    One manifest scan per *dataset*, not per slice: a run plans a key per PUMA and
    resolving them one at a time would be a scan apiece to answer one question.
    Defers to ``load_pipeline.coverage``, so a hit here is a hit for the pipeline.
    """
    root_uri = resolve_root(root_uri)
    covered: list[DatasetSlice] = []
    missing: list[DatasetSlice] = []
    by_dataset: dict[str, list[DatasetSlice]] = {}
    for entry in slices:
        by_dataset.setdefault(entry.dataset_name, []).append(entry)
    for dataset_name, group in by_dataset.items():
        have = coverage.existing_params(dataset_name, root_uri, as_of=as_of)
        for entry in group:
            (covered if entry.key in have else missing).append(entry)
    return Coverage(covered=tuple(covered), missing=tuple(missing))


def read_datasets(
    slices: Sequence[DatasetSlice],
    root_uri: str | pathlib.Path,
    *,
    as_of: dt.datetime | None = None,
    allow_missing: bool = False,
) -> dict[str, pl.DataFrame]:
    """
    Read every slice back, concatenated into one frame per dataset.

    Keyed by dataset name rather than returned as ten named fields: the set of
    datasets is a property of the pipeline, and a caller that wants one asks for
    it by the same constant the pipeline writes it under.

    ``allow_missing`` skips a slice with no write rather than raising -- for the
    caller that already knows some geographies failed. A dataset whose every slice
    is missing is absent from the result rather than present and empty.
    """
    root_uri = resolve_root(root_uri)
    frames: dict[str, list[pl.DataFrame]] = {}
    for entry in slices:
        try:
            frames.setdefault(entry.dataset_name, []).append(
                entry.read(root_uri, as_of)
            )
        except Exception as exc:
            if not allow_missing:
                raise
            logger.warning("skipping %s: %s", entry.label, exc)
    return {
        name: pl.concat(parts, how="vertical_relaxed")
        for name, parts in frames.items()
        if parts
    }


def dataset_files(
    slices: Sequence[DatasetSlice],
    root_uri: str | pathlib.Path,
    *,
    as_of: dt.datetime | None = None,
) -> dict[str, list[str]]:
    """
    The parquet behind each slice, keyed by dataset name.

    Read out of the manifest rather than collected during a run: a reused slice
    was written by an earlier run this one never saw, so the run cannot name it.
    """
    root_uri = resolve_root(root_uri)
    files: dict[str, list[str]] = {}
    by_dataset: dict[str, list[DatasetSlice]] = {}
    for entry in slices:
        by_dataset.setdefault(entry.dataset_name, []).append(entry)
    for dataset_name, group in by_dataset.items():
        have = coverage.existing_params(dataset_name, root_uri, as_of=as_of)
        found = [have[e.key].data_uri for e in group if e.key in have]
        if found:
            files[dataset_name] = sorted(found)
    return files


# --------------------------------------------------------------------------- #
# Geography input
# --------------------------------------------------------------------------- #


def parse_pumas_file(path: str | pathlib.Path) -> list[str]:
    """
    Read a file of PUMA GISJOINs, one per line.

    Blank lines and ``#`` comments are skipped, so a list can carry notes and have
    PUMAs commented out rather than deleted. The codes themselves are validated
    where they are consolidated, which is where a bad one can name its geography.
    """
    codes: list[str] = []
    for line in pathlib.Path(path).read_text().splitlines():
        text = line.split("#", 1)[0].strip()
        if text:
            codes.append(text)
    if not codes:
        msg = f"{path}: no PUMAs found"
        raise ValueError(msg)
    return codes


def pumas_from_args(args: object) -> list[str]:
    """Every PUMA named on the command line, from flags and files alike."""
    codes: list[str] = list(getattr(args, "pumas", None) or [])
    for batch in getattr(args, "pumas_files", None) or []:
        codes.extend(batch)
    return codes
