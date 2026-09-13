"""
Two calls from a list of coordinates to the climate datasets, in memory.

One store, and it is wherever ``root_uri`` points -- a shared drive, typically,
named by ``$CLIMATE_ROOT``. The pipeline writes there and the frames are read
back from there; nothing is staged on the caller's disk.

The two entry points differ only in what they do about a location the store does
not have:

- :func:`fetch` **refuses**. It is the read-only door: whatever comes back was
  already there, and a missing point is an error rather than an hour of
  downloads nobody asked for.
- :func:`fetch_and_ingest` **ingests it**, by running
  ``batch_jobs.climate_pipeline.flows.run_climate_pipeline`` for the missing
  points and then reading everything back.

Separating them makes the expensive one explicit at the call site. A single
function with a flag reads the same either way, and the flag is easy to leave at
whatever the last caller wanted.

The domain half -- consolidating the coordinates, asking the manifests what
exists, reading the frames -- is ``external_data.data_fetch.climate``. This lives
here because running the suite means Prefect, which a package may not import.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from collections.abc import Iterable, Sequence

import common.exceptions
from batch_jobs.climate_pipeline.flows import run_climate_pipeline
from external_data.climate_pipeline import point_manifest, schema
from external_data.data_fetch import climate as data_fetch
from external_data.data_fetch import consolidate
from external_data.data_fetch.climate import Point

logger = logging.getLogger(__name__)


def _prepare(
    points: Iterable[Sequence[float]],
    start_date: dt.date,
    end_date: dt.date,
    root_uri: str,
    nsrdb: schema.NsrdbRequestArgs,
    as_of: dt.datetime | None,
) -> tuple[str, consolidate.ConsolidatedPoints, data_fetch.CoveragePlan]:
    """The work both entry points share: validate, consolidate, ask the store."""
    # Checked before anything else, because a fully-covered request never reaches
    # the flow and would otherwise read the wrong slice without comment.
    if nsrdb.interval != schema.NSRDB_SILVER_INTERVAL:
        msg = (
            f"the silver join reads the {schema.NSRDB_SILVER_INTERVAL}-minute "
            f"slice, but the NSRDB config has interval={nsrdb.interval}"
        )
        raise common.exceptions.PipelineValueError(msg)

    root_uri = data_fetch.resolve_root(root_uri)
    grid = consolidate.consolidate(points)
    plan = data_fetch.plan_coverage(
        grid,
        start_date,
        end_date,
        root_uri,
        as_of=as_of,
    )
    logger.info("coverage at %s: %s", root_uri, data_fetch.describe_plan(plan))
    return root_uri, grid, plan


def _read(
    grid: consolidate.ConsolidatedPoints,
    plan: data_fetch.CoveragePlan,
    start_date: dt.date,
    end_date: dt.date,
    root_uri: str,
    as_of: dt.datetime | None,
    allow_missing: bool,
    ran_pipeline: bool,
) -> data_fetch.ClimateDatasets:
    """Read the silver grid back and assemble the result."""
    silver_frame = data_fetch.read_silver(
        grid,
        start_date,
        end_date,
        root_uri,
        as_of=as_of,
        allow_missing=allow_missing,
    )
    logger.info("climate silver: %d row(s)", silver_frame.height)
    result = data_fetch.ClimateDatasets(
        silver=silver_frame,
        points=grid,
        coverage=plan,
        ran_pipeline=ran_pipeline,
    )
    # Logged here rather than at consolidation, which runs before anything is
    # read and so cannot name the NSRDB cell.
    logger.info("point mapping:")
    for line in result.describe_mapping():
        logger.info("  %s", line)
    return result


def fetch(
    points: Iterable[Sequence[float]],
    *,
    start_date: dt.date = schema.NSRDB_MIN_DATE,
    end_date: dt.date = schema.NSRDB_MAX_DATE,
    root_uri: str | None = None,
    nsrdb: schema.NsrdbRequestArgs | None = None,
    as_of: dt.datetime | None = None,
) -> data_fetch.ClimateDatasets:
    """
    Read the climate datasets for *points* out of the store, without ingesting.

    **Returns what the store has and reports what it does not.** A request for
    five locations where three are present comes back with those three rather
    than nothing: the two absent ones cost the caller a message, not the data
    they already own. Which locations fell short is on the returned
    ``coverage``, per dataset, and is logged as a warning.

    Raises only when the store has nothing at all for the request -- there is no
    partial answer to give, and the caller is asking about a store that has never
    seen these locations.

    Never ingests, whatever is missing. Filling the gaps is
    :func:`fetch_and_ingest`, which is a separate call precisely so that
    answering "is it there?" cannot start an hour of downloads.
    """
    nsrdb = nsrdb if nsrdb is not None else schema.NsrdbRequestArgs()
    root_uri = root_uri if root_uri is not None else data_fetch.default_root()
    root_uri, grid, plan = _prepare(
        points, start_date, end_date, root_uri, nsrdb, as_of
    )

    if not plan.complete:
        missing = _describe_missing(plan)
        if not _can_answer_partially(plan):
            msg = (
                f"{root_uri} has nothing for this request: {missing}. "
                "Use fetch_and_ingest to ingest it."
            )
            raise common.exceptions.PipelineValueError(msg)
        logger.warning(
            "%s does not cover all of this request: %s. Returning the locations "
            "it has; use fetch_and_ingest to fill the gaps.",
            root_uri,
            missing,
        )

    return _read(
        grid,
        plan,
        start_date,
        end_date,
        root_uri,
        as_of,
        # Partial by design: the frame comes back holding the locations the store
        # covers, and ``coverage`` says which ones it does not.
        allow_missing=True,
        ran_pipeline=False,
    )


def _can_answer_partially(plan: data_fetch.CoveragePlan) -> bool:
    """
    Is there anything to return at all?

    With silver the only dataset there is one question to ask, where the
    three-frame result needed every dataset to have something.
    """
    return bool(plan.silver.covered)


def _describe_missing(plan: data_fetch.CoveragePlan) -> str:
    """Which locations the silver grid falls short at."""
    return f"silver missing {point_manifest.describe_points(plan.silver.missing)}"


def _nodes_needing_work(
    grid: consolidate.ConsolidatedPoints, plan: data_fetch.CoveragePlan
) -> tuple[Point, ...]:
    """
    The nodes whose silver is missing, in the caller's order.

    Narrowed because the flow's silver join rebuilds *every* node in the request
    it is handed -- it has no per-node coverage check -- so passing the whole
    request would rebuild silver for nodes that already have it, superseding good
    tables and orphaning their parquet.

    A node needing silver may already have its bronze; the flow's own coverage
    check skips the fetch and goes straight to the join.
    """
    missing = set(plan.silver.missing)
    return tuple(node for node in grid.nodes if node in missing)


def fetch_and_ingest(
    points: Iterable[Sequence[float]],
    *,
    start_date: dt.date = schema.NSRDB_MIN_DATE,
    end_date: dt.date = schema.NSRDB_MAX_DATE,
    root_uri: str | None = None,
    era5: schema.Era5RequestArgs | None = None,
    nsrdb: schema.NsrdbRequestArgs | None = None,
    writer: str = data_fetch.DEFAULT_WRITER,
    force_refresh: bool = False,
    allow_missing_points: bool = False,
    as_of: dt.datetime | None = None,
) -> data_fetch.ClimateDatasets:
    """
    Ingest whatever the store is missing for *points*, then read everything back.

    The pipeline is skipped entirely when the store already covers the request,
    so a repeat call costs nothing. ``force_refresh`` runs it anyway.
    ``allow_missing_points`` reads over locations with no bronze; a location an
    earlier run recorded as permanently unserviceable enables that on its own,
    since such a gap is explained rather than unexpected.
    """
    era5 = era5 if era5 is not None else schema.Era5RequestArgs()
    nsrdb = nsrdb if nsrdb is not None else schema.NsrdbRequestArgs()
    root_uri = root_uri if root_uri is not None else data_fetch.default_root()
    root_uri, grid, plan = _prepare(
        points, start_date, end_date, root_uri, nsrdb, as_of
    )

    ran_pipeline = force_refresh or not plan.complete
    if ran_pipeline:
        # Only the nodes that need work. The flow's silver join rebuilds *every*
        # node in the request it is handed -- it has no per-node coverage check --
        # so passing the whole request would rebuild silver for nodes that
        # already have it, superseding good tables and orphaning their parquet.
        todo = grid.nodes if force_refresh else _nodes_needing_work(grid, plan)
        run_climate_pipeline(
            request=schema.ClimatePipelineRequestArgs(
                points=todo, start_date=start_date, end_date=end_date
            ),
            era5=era5,
            nsrdb=nsrdb,
            root_uri=root_uri,
            writer=writer,
            force_refresh=force_refresh,
            allow_missing_points=allow_missing_points,
            # Deliberately not the caller's ``as_of``. The bronze this run writes
            # is stamped ``write_time = now``, and the flow threads ``as_of``
            # into its own silver step, which resolves ``write_time <= as_of`` --
            # so a past bound makes the join fail to find the bronze the run just
            # paid hours for. ``run_load_pipeline`` guards its silver step the
            # same way for the same reason. A run cannot honour a past bound
            # anyway: it exists to produce data newer than one.
        )
    else:
        logger.info("every requested point is already covered; not ingesting")

    # A location the providers have permanently refused is a recorded gap, not an
    # unexplained one, so reading over it is right -- otherwise a single sea point
    # would fail every future call for the other locations too.
    gaps = data_fetch.recorded_gaps(grid, root_uri)
    if gaps and not allow_missing_points:
        logger.warning(
            "reading over %s recorded permanently unavailable by an earlier run",
            ", ".join(
                f"{source}: {len(points)} point(s)" for source, points in gaps.items()
            ),
        )

    return _read(
        grid,
        plan,
        start_date,
        end_date,
        root_uri,
        # Not the caller's ``as_of`` when this run just wrote: the pipeline stamps
        # ``write_time = now`` and a read resolves ``write_time <= as_of``, so a
        # past ``as_of`` would filter out the very rows the fetch just paid for.
        # ``run_climate_pipeline`` guards its own silver step the same way.
        # Nothing ran means nothing new, so the bound is honoured as asked.
        None if ran_pipeline else as_of,
        allow_missing=allow_missing_points or bool(gaps),
        ran_pipeline=ran_pipeline,
    )


def build_parser() -> argparse.ArgumentParser:
    """The command-line front end to the two entry points."""
    parser = argparse.ArgumentParser(
        prog="python -m batch_jobs.data_fetch.climate",
        description=(
            "Read the ERA5-Land, NSRDB and silver climate datasets for a set of "
            "coordinates out of the store, ingesting first with --ingest."
        ),
    )
    parser.add_argument(
        "-p",
        "--point",
        dest="points",
        action="append",
        type=data_fetch.parse_point,
        metavar="LAT,LON",
        help="coordinate in decimal degrees; repeat the flag for several points",
    )
    parser.add_argument(
        "-f",
        "--points-file",
        dest="points_files",
        action="append",
        type=data_fetch.parse_points_file,
        metavar="PATH",
        help=(
            "file of 'lat,lon' lines ('#' comments and blank lines ignored); "
            "repeatable, and combines with --point"
        ),
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help=(
            "ingest the locations the store does not have, rather than failing on them"
        ),
    )
    parser.add_argument(
        "--start-date",
        type=dt.date.fromisoformat,
        default=schema.NSRDB_MIN_DATE,
        metavar="YYYY-MM-DD",
        help="first day to cover (default: %(default)s, the start of NSRDB cover)",
    )
    parser.add_argument(
        "--end-date",
        type=dt.date.fromisoformat,
        default=schema.NSRDB_MAX_DATE,
        metavar="YYYY-MM-DD",
        help="last day to cover (default: %(default)s, the end of NSRDB cover)",
    )
    parser.add_argument(
        "--root-uri",
        default=None,
        help=(
            "the store to read and write; a relative path resolves against the "
            f"working directory (default: ${data_fetch.ROOT_ENV}, else "
            f"{data_fetch.DEFAULT_ROOT})"
        ),
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="with --ingest, run the pipeline even where the manifest has coverage",
    )
    parser.add_argument(
        "--allow-missing-points",
        action="store_true",
        help="with --ingest, read over points with no bronze instead of failing",
    )
    return parser


if __name__ == "__main__":
    _parser = build_parser()
    _args = _parser.parse_args()
    _points = data_fetch.points_from_args(_args)
    if not _points:
        # Checked here rather than with ``required``: either flag satisfies it,
        # which argparse cannot express on a single argument.
        _parser.error("at least one --point or --points-file is required")

    # Importing the flow module pulls in Prefect, which installs its own root
    # handler -- adding a second unconditionally prints every line twice.
    _root = logging.getLogger()
    if not _root.handlers:
        _handler = logging.StreamHandler()
        _handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        _root.addHandler(_handler)
    for _name in ("external_data", "batch_jobs", __name__):
        logging.getLogger(_name).setLevel(logging.INFO)

    if _args.ingest:
        _result = fetch_and_ingest(
            _points,
            start_date=_args.start_date,
            end_date=_args.end_date,
            root_uri=_args.root_uri,
            force_refresh=_args.force_refresh,
            allow_missing_points=_args.allow_missing_points,
        )
    else:
        _result = fetch(
            _points,
            start_date=_args.start_date,
            end_date=_args.end_date,
            root_uri=_args.root_uri,
        )

    _root_uri = data_fetch.resolve_root(_args.root_uri or data_fetch.default_root())
    print(f"\nroot: {_root_uri}")
    print(f"ingested: {'yes' if _result.ran_pipeline else 'no, already covered'}")
    print(f"  silver  {_result.silver.shape}")
    print("\nfiles:")
    for _uri in data_fetch.dataset_files(
        _result, _args.start_date, _args.end_date, _root_uri
    ):
        print(f"  {_uri}")
