"""
The domain half of the climate data-fetch helper: what already exists, and how to
read it back.

Deliberately **not** the orchestration. Running the suite means the concurrent
fan-out in ``batch_jobs.climate_pipeline.flows.run_climate_pipeline``, and a
package may not import Prefect or ``batch_jobs``, so the entry point that ties
these together lives in ``batch_jobs.data_fetch.climate`` and calls into here.

What is here:

- :func:`plan_coverage` -- what the manifests already hold for a set of points,
  per dataset. Its job is to answer "is there anything to do?" *before* a run,
  which is what lets a fully-covered request skip the pipeline entirely.
- :func:`read_frames` -- the three datasets read back into memory.
- :func:`dataset_files` -- the parquet behind each of them.
- the coordinate parsing the command line needs.

All of it is manifest reads and parquet reads: no fetching, no writing.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import logging
import os
import pathlib
from collections.abc import Callable, Sequence

import polars as pl
import pydantic

import common.exceptions
from common.storage import columnar
from common.storage.columnar import BaseDataFrameSchema
from common.storage.manifest import ManifestRow
from external_data.climate_pipeline import (
    dead_letter,
    point_manifest,
    schema,
    silver,
)
from external_data.climate_pipeline.era5 import bronze as era5_bronze
from external_data.data_fetch import consolidate

logger = logging.getLogger(__name__)

Point = point_manifest.Point

# Relative fallback for an ad-hoc call, matching the flows' own default; a caller
# with somewhere to put the data passes ``root_uri``.
DEFAULT_ROOT = "climate_pipeline"
DEFAULT_WRITER = "climate_data_fetch"

# All three datasets are timeseries keyed on this column, which is what makes one
# trimming rule enough for each of them.
_TIME_COLUMN = "valid_time"

# The store every run reads and writes, named by environment rather than
# hard-coded: the path is per-machine (a mounted drive, a sync folder) and baking
# one in would embed one engineer's home directory in the repo.
ROOT_ENV = "CLIMATE_ROOT"


def default_root() -> str:
    """The store named in the environment, or the relative fallback."""
    return os.environ.get(ROOT_ENV) or DEFAULT_ROOT


@dataclasses.dataclass(frozen=True, kw_only=True)
class Coverage:
    """
    What one dataset already holds for the requested points.

    Points are reported at the grain the dataset is *stored* at -- grid nodes for
    ERA5 and silver, exact coordinates for NSRDB -- because that is the grain a
    lookup can answer at. Reporting the caller's own coordinates would suggest a
    precision the manifest does not have.
    """

    covered: tuple[Point, ...]
    missing: tuple[Point, ...]

    @property
    def complete(self) -> bool:
        return not self.missing


@dataclasses.dataclass(frozen=True, kw_only=True)
class CoveragePlan:
    """
    What the store already holds for a request, read before anything runs.

    Only the silver grid is planned. The ERA5 and NSRDB bronze it is joined from
    are *inputs*: a node whose silver exists needs neither read, and a node whose
    silver is missing gets both fetched by the pipeline behind its own coverage
    check. Planning them here would ask the manifest two questions whose answers
    change nothing.

    ``complete`` is the question the pre-check exists to answer: when it is true
    there is nothing to run, and the caller can go straight to reading the frame.
    """

    silver: Coverage
    # The window the solar actually covers, once the request has been clamped to
    # NSRDB's coverage. ``None`` means the range is entirely outside it, and the
    # silver grid carries null solar for those hours.
    nsrdb_range: tuple[dt.date, dt.date] | None

    @property
    def complete(self) -> bool:
        return self.silver.complete


@dataclasses.dataclass(frozen=True, kw_only=True)
class ClimateDatasets:
    """
    The silver grid for a request, in memory.

    Silver only: the ERA5 and NSRDB bronze it is joined from are inputs to that
    join, and a caller asking for climate data wants the harmonised hourly grid
    rather than the two half-tables it was built from. The bronze is still
    ingested when the silver has to be built -- it is simply not planned,
    fetched or returned.

    ``coverage`` is what the store held *before* the call, and ``ran_pipeline``
    whether the pipeline was invoked -- always ``False`` from :func:`fetch`.
    """

    silver: pl.DataFrame
    points: consolidate.ConsolidatedPoints
    coverage: CoveragePlan
    ran_pipeline: bool

    def rows_for(
        self, point: Sequence[float], *, with_node: bool = False
    ) -> pl.DataFrame:
        """
        The silver rows for one coordinate **as it was requested**.

        The frame is keyed by the 0.1 deg node, never by the coordinate a caller
        typed, so slicing it by hand means knowing the snapping rule. This does
        it instead.

        ``with_node`` adds ``node_latitude``/``node_longitude``, so several
        points' frames can be concatenated and still say which is which. Opt-in:
        the extra columns make the frame fail ``Era5NsrdbSilverSchema.validate``
        and stop it matching the parquet it came from, both worth keeping by
        default.
        """
        key = point_manifest.normalise(point)
        node = self.points.node_by_input.get(key)
        if node is None:
            requested = point_manifest.describe_points(self.points.node_by_input)
            msg = (
                f"{key} was not part of this request; it covered {requested}. "
                "Coordinates are matched at "
                f"{point_manifest.POINT_DECIMALS} decimal places."
            )
            raise common.exceptions.PipelineValueError(msg)

        rows = self.silver.filter(
            (pl.col("era5_latitude") == node[0]) & (pl.col("era5_longitude") == node[1])
        )
        if with_node:
            rows = rows.with_columns(
                pl.lit(node[0], dtype=pl.Float64).alias("node_latitude"),
                pl.lit(node[1], dtype=pl.Float64).alias("node_longitude"),
            )
        return rows

    def nsrdb_cell_by_node(self) -> dict[Point, Point]:
        """
        The NSRDB cell each node's data came from, read out of the silver grid.

        Silver is authoritative about it: ``silver._snap_to_nearest_node`` chose
        the cell, keeping the one nearest each node centre and discarding any
        others in the same node. It is also the only source now that the bronze
        is neither fetched nor returned -- which is fine, because the cell is a
        column of the joined table.
        """
        cells: dict[Point, Point] = {}
        if not self.silver.height:
            return cells
        rows = self.silver.select(
            "era5_latitude", "era5_longitude", "nsrdb_latitude", "nsrdb_longitude"
        ).unique()
        for node_lat, node_lon, cell_lat, cell_lon in rows.rows():
            if cell_lat is None or cell_lon is None:
                # ERA5-only silver: no solar covers that node/hour.
                continue
            cells[(node_lat, node_lon)] = (cell_lat, cell_lon)
        return cells

    def describe_mapping(self) -> list[str]:
        """
        One line per node: the coordinates that collapsed onto it, the 0.1 deg
        node ERA5 and silver are keyed by, and the NSRDB cell the data came from.

        Distinct from :func:`consolidate.describe_mapping`, which runs before
        anything is read and so cannot name the cell. That coordinate is NSRDB's
        own, on its ~4 km grid, and is neither the requested coordinate nor the
        node -- so it is the one thing a caller cannot work out for themselves,
        and the reason indexing the NSRDB frame by either of the other two fails.

        Reported exactly as stored, unrounded: the point of naming it is that it
        matches the frame, and rounding would break that wherever a cell centre
        carries more precision than the examples we have seen.
        """
        cells = self.nsrdb_cell_by_node()
        lines: list[str] = []
        for node, points in consolidate.group_by_node(self.points).items():
            rendered = ", ".join(
                f"({_coord(lat)}, {_coord(lon)})" for lat, lon in points
            )
            node_label = f"({node[0]:.1f}, {node[1]:.1f})"
            if node in cells:
                cell = cells[node]
                solar = f"NSRDB ({_coord(cell[0])}, {_coord(cell[1])})"
            elif self.coverage.nsrdb_range is None:
                # The request lies outside NSRDB's window, so those hours carry
                # null solar by design rather than by omission.
                solar = "NSRDB none (outside coverage)"
            else:
                solar = "NSRDB missing"
            lines.append(f"{rendered} -> ERA5/silver {node_label}, {solar}")
        return lines


def _coord(value: float) -> str:
    """
    A coordinate rendered without losing digits.

    ``%g`` counts *significant* digits, so it silently truncates a longitude:
    ``-122.4233`` prints as ``-122.423``, which is a different place and, worse,
    a key that indexes nothing. Coordinates here are stored to
    ``point_manifest.POINT_DECIMALS``, so render that many and trim the padding.
    """
    rendered = f"{value:.{point_manifest.POINT_DECIMALS}f}".rstrip("0")
    # A whole number keeps one decimal: "-122" reads like a different place.
    return f"{rendered}0" if rendered.endswith(".") else rendered


def resolve_root(root_uri: str | pathlib.Path) -> str:
    """A relative root has no scheme; resolve it so the manifest readers and the
    flow agree on which store is meant (mirrors the ingest helpers)."""
    if "://" not in str(root_uri):
        return str(pathlib.Path(root_uri).resolve())
    return str(root_uri)


def _coverage(
    points: Sequence[Point],
    dataset_name: str,
    root_uri: str,
    start_date: dt.date,
    end_date: dt.date,
    *,
    grain: point_manifest.Grain = point_manifest.normalise,
    as_of: dt.datetime | None = None,
    match: Callable[[ManifestRow], bool] | None = None,
) -> Coverage:
    """One dataset's coverage, in a single manifest scan."""
    rows, missing = point_manifest.covered_rows(
        points,
        dataset_name,
        root_uri,
        point_manifest.read_point_key,
        start_date,
        end_date,
        grain=grain,
        as_of=as_of,
        match=match,
    )
    return Coverage(covered=tuple(sorted(rows)), missing=tuple(sorted(missing)))


def plan_coverage(
    grid: consolidate.ConsolidatedPoints,
    start_date: dt.date,
    end_date: dt.date,
    root_uri: str | pathlib.Path,
    *,
    as_of: dt.datetime | None = None,
) -> CoveragePlan:
    """
    What the store already holds of the silver grid for *grid*.

    Read before running anything, so a request whose every node is already built
    can skip the pipeline rather than pay for a run that would rebuild what it
    has.

    Looked up at the grid node, which is what ``silver.write_silver`` writes at.
    The bronze is not consulted: silver is what the caller gets, and a node whose
    silver exists needs no bronze read to prove it.
    """
    root_uri = resolve_root(root_uri)
    silver_coverage = _coverage(
        grid.nodes,
        silver.DATASET_NAME,
        root_uri,
        start_date,
        end_date,
        grain=era5_bronze.node,
        as_of=as_of,
    )
    # Derived, not looked up: the clamp says which hours can carry solar at all,
    # which is what makes a null in the silver grid expected rather than a gap.
    nreq = silver.nsrdb_request(
        schema.ClimatePipelineRequestArgs(
            points=grid.nodes, start_date=start_date, end_date=end_date
        )
    )
    nsrdb_range = None if nreq is None else (nreq.start_date, nreq.end_date)
    return CoveragePlan(silver=silver_coverage, nsrdb_range=nsrdb_range)


def describe_plan(plan: CoveragePlan) -> str:
    """A one-line summary of a coverage plan, for a log or a terminal."""
    return (
        f"silver {len(plan.silver.covered)} covered/{len(plan.silver.missing)} missing"
    )


def recorded_gaps(
    grid: consolidate.ConsolidatedPoints, root_uri: str | pathlib.Path
) -> dict[str, tuple[Point, ...]]:
    """
    Requested points an earlier run recorded as permanently unserviceable.

    A provider that has refused a point will refuse it again, so reading strictly
    over that gap would fail every run for data that is never going to arrive.
    Distinct from a gap nothing explains -- that usually means a step was never
    run, and should still fail.

    Only the pipeline flow writes these records, into its flow manifest. Reading
    them is what lets a later call tell the two kinds of gap apart.
    """
    root_uri = resolve_root(root_uri)
    era5_dead = dead_letter.known_points(root_uri, "era5", era5_bronze.node)
    nsrdb_dead = dead_letter.known_points(root_uri, "nsrdb", point_manifest.normalise)
    gaps = {
        "era5": tuple(
            sorted({era5_bronze.node(p) for p in grid.era5_points} & era5_dead.keys())
        ),
        "nsrdb": tuple(sorted(set(grid.nsrdb_points) & nsrdb_dead.keys())),
    }
    return {source: points for source, points in gaps.items() if points}


def _read_point(
    dataset_name: str,
    schema_cls: type[BaseDataFrameSchema],
    key: pydantic.BaseModel,
    row: ManifestRow,
    root_uri: str,
    as_of: dt.datetime | None,
) -> pl.DataFrame:
    """
    One point's table, validated against its schema where that is possible.

    ``read_dataset`` resolves on exact ``params_json`` equality, so it finds a
    write only when the stored key *is* the key asked for. That is the common
    case -- a run writes what the request asked for -- and it is worth taking,
    since it validates the frame against the dataset's schema on the way in.

    A stored write spanning more than the request does not match, and coverage
    here is containment rather than equality: a wider write legitimately covers a
    narrower ask. So the manifest row the planner already resolved is the
    fallback, read straight from its ``data_uri``.
    """
    try:
        return columnar.read_dataset(
            schema_cls, dataset_name, key, root_uri, as_of=as_of
        )
    except KeyError:
        # The planner found this row by containment; the key it was written under
        # simply spans more than was asked for.
        return pl.read_parquet(row.data_uri)


def read_silver(
    grid: consolidate.ConsolidatedPoints,
    start_date: dt.date,
    end_date: dt.date,
    root_uri: str | pathlib.Path,
    *,
    as_of: dt.datetime | None = None,
    allow_missing: bool = False,
) -> pl.DataFrame:
    """
    Read the silver grid back into memory.

    One entry per node, so each is read on its own and the parts concatenated --
    there is no batch key to resolve.

    ``allow_missing`` skips a node with no write rather than raising; off by
    default, since a frame quietly covering fewer locations is worse than an
    error.
    """
    root_uri = resolve_root(root_uri)
    return _read_dataset_points(
        silver.DATASET_NAME,
        silver.Era5NsrdbSilverSchema,
        grid.nodes,
        lambda point: silver.node_params(
            era5_bronze.node(point),
            schema.ClimatePipelineRequestArgs(
                points=(point,), start_date=start_date, end_date=end_date
            ),
        ),
        era5_bronze.node,
        root_uri,
        start_date,
        end_date,
        as_of,
        allow_missing,
        silver.KEY_COLUMNS,
    )


def _read_dataset_points(
    dataset_name: str,
    schema_cls: type[BaseDataFrameSchema],
    points: Sequence[Point],
    key_for: Callable[[Point], pydantic.BaseModel],
    grain: point_manifest.Grain,
    root_uri: str,
    start_date: dt.date,
    end_date: dt.date,
    as_of: dt.datetime | None,
    allow_missing: bool,
    key_columns: Sequence[str],
    match: Callable[[ManifestRow], bool] | None = None,
) -> pl.DataFrame:
    """Every requested point's table for one dataset, concatenated and deduped."""
    rows, missing = point_manifest.covered_rows(
        points,
        dataset_name,
        root_uri,
        point_manifest.read_point_key,
        start_date,
        end_date,
        grain=grain,
        as_of=as_of,
        match=match,
    )
    if missing and not allow_missing:
        raise common.exceptions.PipelineValueError(
            f"no {dataset_name} covering {len(missing)} of "
            f"{len(missing) + len(rows)} requested location(s) for "
            f"[{start_date}, {end_date}] under {root_uri}: "
            f"{point_manifest.describe_points(missing)}"
        )
    if missing:
        logger.warning(
            "%s is missing %s; continuing without them",
            dataset_name,
            point_manifest.describe_points(missing),
        )
    if not rows:
        raise common.exceptions.PipelineValueError(
            f"no {dataset_name} found under {root_uri} for any requested location"
        )
    frames = [
        _read_point(dataset_name, schema_cls, key_for(point), row, root_uri, as_of)
        for point, row in rows.items()
    ]
    return (
        pl.concat(frames, how="vertical_relaxed")
        # Trim to the days asked for. Coverage is containment -- a write spanning
        # 2020..2024 legitimately covers a request for 2020..2021 -- so a stored
        # write routinely holds more than the request, and handing back the whole
        # file would answer a question nobody asked.
        .filter(pl.col(_TIME_COLUMN).dt.date().is_between(start_date, end_date))
        .unique(subset=list(key_columns))
        .sort(list(key_columns))
    )


def dataset_files(
    result: ClimateDatasets,
    start_date: dt.date,
    end_date: dt.date,
    root_uri: str | pathlib.Path,
    as_of: dt.datetime | None = None,
) -> list[str]:
    """
    The parquet behind the silver grid in *result*.

    Read out of the manifest rather than collected during a run: a reused node's
    file was written by an earlier run this one never saw, so the run cannot name
    it. The manifest is the index for exactly this reason.
    """
    root_uri = resolve_root(root_uri)
    rows, _missing = point_manifest.covered_rows(
        result.points.nodes,
        silver.DATASET_NAME,
        root_uri,
        point_manifest.read_point_key,
        start_date,
        end_date,
        grain=era5_bronze.node,
        as_of=as_of,
    )
    return sorted(row.data_uri for row in rows.values())


# --------------------------------------------------------------------------- #
# Coordinate input
# --------------------------------------------------------------------------- #


def parse_point(value: str) -> Point:
    """
    Parse a ``lat,lon`` string into a point.

    Its own function so a malformed coordinate is rejected with the text that
    caused it, rather than failing somewhere inside a fetch with the original
    input long out of view.
    """
    parts = value.split(",")
    if len(parts) != 2:
        msg = f"expected 'lat,lon', got {value!r}"
        raise argparse.ArgumentTypeError(msg)
    try:
        return (float(parts[0]), float(parts[1]))
    except ValueError:
        msg = f"expected two numbers in 'lat,lon', got {value!r}"
        raise argparse.ArgumentTypeError(msg) from None


def parse_points_file(path: str | pathlib.Path) -> list[Point]:
    """
    Read a file of ``lat,lon`` lines into points.

    Blank lines and ``#`` comments are skipped, so a list can carry notes and
    have points commented out rather than deleted. A bad line reports its own
    number: in a file of several hundred coordinates, "expected 'lat,lon'" with
    nothing saying where is not something a caller can act on.
    """
    points: list[Point] = []
    lines = pathlib.Path(path).read_text().splitlines()
    for number, line in enumerate(lines, start=1):
        text = line.split("#", 1)[0].strip()
        if not text:
            continue
        try:
            points.append(parse_point(text))
        except argparse.ArgumentTypeError as exc:
            msg = f"{path}: line {number}: {exc}"
            raise argparse.ArgumentTypeError(msg) from None
    if not points:
        msg = f"{path}: no points found"
        raise argparse.ArgumentTypeError(msg)
    return points


def points_from_args(args: argparse.Namespace) -> list[Point]:
    """
    Every point named on the command line, from ``--point`` and files alike.

    Order is preserved and duplicates are left in: consolidation is the fetch's
    job, and silently de-duplicating here would hide a list that repeats itself.
    """
    points: list[Point] = list(args.points or [])
    for batch in args.points_files or []:
        points.extend(batch)
    return points
