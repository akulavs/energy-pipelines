"""
Two calls from a list of PUMAs to the load datasets, in memory.

One store, and it is wherever ``root_uri`` points -- a shared drive, typically,
named by ``$LOAD_ROOT``. The pipeline writes there and the frames are read back
from there; nothing is staged on the caller's disk.

The two entry points differ only in what they do about a geography the store does
not have:

- :func:`fetch` **reads what is there and reports what is not**. A request for
  five PUMAs where three are present comes back with those three; the missing
  two cost a message, not the data the caller already owns. It never ingests,
  whatever is missing.
- :func:`fetch_and_ingest` **ingests it**, by running
  ``batch_jobs.load_pipeline.flows.run_load_pipeline`` for what is missing and
  then reading everything back.

Separating them makes the expensive one explicit at the call site: a building
stock ingest is a per-building fan-out over OEDI, and answering "is it there?"
should not start one.

The domain half -- consolidating the geographies, asking the manifests what
exists, reading the frames -- is ``external_data.data_fetch.load``. This lives
here because running the suite means Prefect, which a package may not import.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import logging
from collections.abc import Iterable, Sequence

import polars as pl

import common.exceptions
from batch_jobs.load_pipeline.flows import run_load_pipeline
from external_data.data_fetch import geographies
from external_data.data_fetch import load as data_fetch
from external_data.load_pipeline import schema, silver
from external_data.load_pipeline.comstock import bronze as comstock_bronze
from external_data.load_pipeline.resstock import bronze as resstock_bronze

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, kw_only=True)
class PumaFrames:
    """One requested PUMA's slice of each dataset."""

    resstock_metadata: pl.DataFrame
    resstock_timeseries: pl.DataFrame
    comstock_metadata: pl.DataFrame
    comstock_timeseries: pl.DataFrame
    industrial_metadata: pl.DataFrame
    industrial_timeseries: pl.DataFrame


@dataclasses.dataclass(frozen=True, kw_only=True)
class LoadDatasets:
    """
    Every dataset the load suite returns for a request, in memory.

    Six named frames rather than a dict keyed by dataset name: a caller writing
    ``result.resstock_timeseries`` gets an attribute error at the point of the
    typo, where a dict key gives a ``KeyError`` at read time and only if that
    branch runs. The dsgrid bronze the industrial silver is built from is not
    among them -- it is an input to the join, and a caller wanting industrial
    load wants the joined table.

    ``coverage`` is what the store held *before* the call and ``ran_pipeline``
    whether the pipeline was invoked -- always ``False`` from :func:`fetch`.
    """

    resstock_metadata: pl.DataFrame
    resstock_timeseries: pl.DataFrame
    comstock_metadata: pl.DataFrame
    comstock_timeseries: pl.DataFrame
    industrial_metadata: pl.DataFrame
    industrial_timeseries: pl.DataFrame
    geographies: geographies.ConsolidatedGeographies
    slices: tuple[data_fetch.DatasetSlice, ...]
    coverage: data_fetch.Coverage
    ran_pipeline: bool

    def rows_for(self, puma_gisjoin: str) -> PumaFrames:
        """
        Every dataset's rows for one requested PUMA.

        The six are keyed at two grains -- the building-stock ones per PUMA, the
        industrial silver per state, since dsgrid publishes per state and covers
        every county at once -- so slicing by hand means knowing which is which.
        This filters each frame by whichever column it actually carries.
        """
        state = self.geographies.state_by_puma.get(puma_gisjoin)
        if state is None:
            known = ", ".join(sorted(self.geographies.state_by_puma))
            msg = f"{puma_gisjoin} was not part of this request; it covered {known}."
            raise common.exceptions.PipelineValueError(msg)

        def _slice(frame: pl.DataFrame) -> pl.DataFrame:
            if "puma_gisjoin" in frame.columns:
                return frame.filter(pl.col("puma_gisjoin") == puma_gisjoin)
            if "state" in frame.columns:
                return frame.filter(pl.col("state") == state)
            # Keyed by neither -- a county-grained industrial table, whose rows
            # all belong to the state that was asked for anyway.
            return frame

        return PumaFrames(
            resstock_metadata=_slice(self.resstock_metadata),
            resstock_timeseries=_slice(self.resstock_timeseries),
            comstock_metadata=_slice(self.comstock_metadata),
            comstock_timeseries=_slice(self.comstock_timeseries),
            industrial_metadata=_slice(self.industrial_metadata),
            industrial_timeseries=_slice(self.industrial_timeseries),
        )

    def describe_mapping(self) -> list[str]:
        """One line per state, listing the PUMAs that resolve through it."""
        by_state: dict[str, list[str]] = {}
        for puma, state in self.geographies.state_by_puma.items():
            by_state.setdefault(state, []).append(puma)
        return [
            f"{', '.join(sorted(pumas))} -> state {state} (industrial silver)"
            for state, pumas in sorted(by_state.items())
        ]


# Which named field each dataset's frame lands in. One place, so a dataset name
# and the attribute it becomes cannot drift apart.
_FIELD_BY_DATASET = {
    resstock_bronze.METADATA_DATASET_NAME: "resstock_metadata",
    resstock_bronze.TIMESERIES_DATASET_NAME: "resstock_timeseries",
    comstock_bronze.PUMA_METADATA_DATASET_NAME: "comstock_metadata",
    comstock_bronze.TIMESERIES_DATASET_NAME: "comstock_timeseries",
    silver.METADATA_DATASET_NAME: "industrial_metadata",
    silver.TIMESERIES_DATASET_NAME: "industrial_timeseries",
}


def _prepare(
    pumas: Iterable[str] | schema.LoadGeographies,
    root_uri: str,
    resstock: schema.ResstockRequestArgs,
    comstock: schema.ComstockRequestArgs,
    dsgrid: schema.DsgridRequestArgs,
    as_of: dt.datetime | None,
) -> tuple[
    str,
    geographies.ConsolidatedGeographies,
    list[data_fetch.DatasetSlice],
    data_fetch.Coverage,
]:
    """The work both entry points share: consolidate, plan, ask the store."""
    root_uri = data_fetch.resolve_root(root_uri)
    grid = geographies.consolidate(pumas)
    slices = data_fetch.plan_slices(
        grid, resstock=resstock, comstock=comstock, dsgrid=dsgrid
    )
    cover = data_fetch.plan_coverage(slices, root_uri, as_of=as_of)
    logger.info("coverage at %s: %s", root_uri, data_fetch.describe_coverage(cover))
    return root_uri, grid, slices, cover


def _read(
    grid: geographies.ConsolidatedGeographies,
    slices: Sequence[data_fetch.DatasetSlice],
    cover: data_fetch.Coverage,
    root_uri: str,
    as_of: dt.datetime | None,
    allow_missing: bool,
    ran_pipeline: bool,
) -> LoadDatasets:
    """Read every dataset back and assemble the result."""
    frames = data_fetch.read_datasets(
        slices, root_uri, as_of=as_of, allow_missing=allow_missing
    )
    logger.info(
        "load datasets: %s",
        ", ".join(
            f"{name} {frame.height} row(s)" for name, frame in sorted(frames.items())
        ),
    )
    empty = pl.DataFrame()
    result = LoadDatasets(
        **{
            field: frames.get(dataset, empty)
            for dataset, field in _FIELD_BY_DATASET.items()
        },
        geographies=grid,
        slices=tuple(slices),
        coverage=cover,
        ran_pipeline=ran_pipeline,
    )
    logger.info("geography mapping:")
    for line in result.describe_mapping():
        logger.info("  %s", line)
    return result


def _describe_missing(cover: data_fetch.Coverage) -> str:
    """Which datasets fall short, and how many slices each."""
    counts: dict[str, int] = {}
    for entry in cover.missing:
        counts[entry.dataset_name] = counts.get(entry.dataset_name, 0) + 1
    return "; ".join(
        f"{name} missing {n} slice(s)" for name, n in sorted(counts.items())
    )


def fetch(
    pumas: Iterable[str] | schema.LoadGeographies,
    *,
    root_uri: str | None = None,
    resstock: schema.ResstockRequestArgs | None = None,
    comstock: schema.ComstockRequestArgs | None = None,
    dsgrid: schema.DsgridRequestArgs | None = None,
    as_of: dt.datetime | None = None,
) -> LoadDatasets:
    """
    Read the load datasets for *pumas* out of the store, without ingesting.

    Returns what the store has and reports what it does not, so a request for
    five PUMAs where three are present comes back with those three. Which slices
    fell short is on the returned ``coverage`` and is logged as a warning.

    Raises only when the store has nothing at all for the request -- there is no
    partial answer to give.

    Never ingests, whatever is missing. Filling the gaps is
    :func:`fetch_and_ingest`, which is a separate call precisely so that
    answering "is it there?" cannot start a per-building fan-out over OEDI.
    """
    resstock = resstock or schema.ResstockRequestArgs()
    comstock = comstock or schema.ComstockRequestArgs()
    dsgrid = dsgrid or schema.DsgridRequestArgs()
    root_uri = root_uri if root_uri is not None else data_fetch.default_root()
    root_uri, grid, slices, cover = _prepare(
        pumas, root_uri, resstock, comstock, dsgrid, as_of
    )

    if not cover.complete:
        missing = _describe_missing(cover)
        if not cover.covered:
            msg = (
                f"{root_uri} has nothing for this request: {missing}. "
                "Use fetch_and_ingest to ingest it."
            )
            raise common.exceptions.PipelineValueError(msg)
        logger.warning(
            "%s does not cover all of this request: %s. Returning what it has; "
            "use fetch_and_ingest to fill the gaps.",
            root_uri,
            missing,
        )

    return _read(
        grid, slices, cover, root_uri, as_of, allow_missing=True, ran_pipeline=False
    )


def _pumas_needing_work(
    grid: geographies.ConsolidatedGeographies, cover: data_fetch.Coverage
) -> schema.LoadGeographies:
    """
    The PUMAs some dataset is missing, in the caller's order.

    A state-keyed slice belongs to every PUMA of that state, so a missing dsgrid
    or silver slice pulls in one PUMA to carry it -- the flow takes PUMAs, and
    derives the states from them.

    Narrowed because ``industrial_load_silver`` rebuilds both silver tables for
    *every state in the request* -- it has no per-state coverage check -- so
    handing it the whole request would rebuild silver a state already has,
    superseding good tables and orphaning their parquet.
    """
    wanted: set[str] = set()
    states_needed: set[str] = set()
    for entry in cover.missing:
        params = entry.params.model_dump()
        puma = params.get("puma_gisjoin")
        if puma:
            wanted.add(str(puma))
        elif params.get("state"):
            states_needed.add(str(params["state"]))
    for state in states_needed:
        # Any one PUMA of the state carries it; the first keeps the order stable.
        for geography in grid.pumas:
            if geography.state == state:
                wanted.add(geography.puma_gisjoin)
                break
    ordered = [g for g in grid.pumas if g.puma_gisjoin in wanted]
    return schema.LoadGeographies(pumas=tuple(ordered))


def fetch_and_ingest(
    pumas: Iterable[str] | schema.LoadGeographies,
    *,
    root_uri: str | None = None,
    resstock: schema.ResstockRequestArgs | None = None,
    comstock: schema.ComstockRequestArgs | None = None,
    dsgrid: schema.DsgridRequestArgs | None = None,
    writer: str = data_fetch.DEFAULT_WRITER,
    force_refresh: bool = False,
    allow_missing: bool = False,
    as_of: dt.datetime | None = None,
) -> LoadDatasets:
    """
    Ingest whatever the store is missing for *pumas*, then read everything back.

    The pipeline is skipped entirely when the store already covers the request,
    so a repeat call costs nothing. ``force_refresh`` runs it anyway.
    """
    resstock = resstock or schema.ResstockRequestArgs()
    comstock = comstock or schema.ComstockRequestArgs()
    dsgrid = dsgrid or schema.DsgridRequestArgs()
    root_uri = root_uri if root_uri is not None else data_fetch.default_root()
    root_uri, grid, slices, cover = _prepare(
        pumas, root_uri, resstock, comstock, dsgrid, as_of
    )

    ran_pipeline = force_refresh or not cover.complete
    if ran_pipeline:
        todo = grid.geographies if force_refresh else _pumas_needing_work(grid, cover)
        run_load_pipeline(
            geographies=todo,
            resstock=resstock,
            comstock=comstock,
            dsgrid=dsgrid,
            force_refresh=force_refresh,
            root_uri=root_uri,
            writer=writer,
            # Deliberately not the caller's ``as_of``. This run writes with
            # ``write_time = now``, and a read resolves ``write_time <= as_of``,
            # so a past bound would exclude the run's own output -- which
            # ``run_load_pipeline`` already guards its silver step against, for
            # the same reason. A run cannot honour a past bound anyway: it exists
            # to produce data newer than one.
        )
    else:
        logger.info("every requested slice is already covered; not ingesting")

    return _read(
        grid,
        slices,
        cover,
        root_uri,
        # Same reason: what this run just wrote is newer than any past bound.
        # Nothing ran means nothing new, so the bound is honoured as asked.
        None if ran_pipeline else as_of,
        allow_missing=allow_missing,
        ran_pipeline=ran_pipeline,
    )


def build_parser() -> argparse.ArgumentParser:
    """The command-line front end to the two entry points."""
    parser = argparse.ArgumentParser(
        prog="python -m batch_jobs.data_fetch.load",
        description=(
            "Read the ResStock, ComStock, dsgrid and industrial silver datasets "
            "for a set of PUMAs out of the store, ingesting first with --ingest."
        ),
    )
    parser.add_argument(
        "-p",
        "--puma",
        dest="pumas",
        action="append",
        metavar="GISJOIN",
        help="PUMA GISJOIN (e.g. G11000101); repeat the flag for several",
    )
    parser.add_argument(
        "-f",
        "--pumas-file",
        dest="pumas_files",
        action="append",
        type=data_fetch.parse_pumas_file,
        metavar="PATH",
        help=(
            "file of PUMA GISJOINs, one per line ('#' comments and blank lines "
            "ignored); repeatable, and combines with --puma"
        ),
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="ingest the PUMAs the store does not have, rather than reporting them",
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
        "--allow-missing",
        action="store_true",
        help="read over slices with no write instead of failing",
    )
    return parser


if __name__ == "__main__":
    _parser = build_parser()
    _args = _parser.parse_args()
    _pumas = data_fetch.pumas_from_args(_args)
    if not _pumas:
        # Checked here rather than with ``required``: either flag satisfies it,
        # which argparse cannot express on a single argument.
        _parser.error("at least one --puma or --pumas-file is required")

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
            _pumas,
            root_uri=_args.root_uri,
            force_refresh=_args.force_refresh,
            allow_missing=_args.allow_missing,
        )
    else:
        _result = fetch(_pumas, root_uri=_args.root_uri)

    _root_uri = data_fetch.resolve_root(_args.root_uri or data_fetch.default_root())
    print(f"\nroot: {_root_uri}")
    print(f"ingested: {'yes' if _result.ran_pipeline else 'no, already covered'}")
    print("\ndatasets:")
    for _field in _FIELD_BY_DATASET.values():
        print(f"  {_field:24} {getattr(_result, _field).shape}")
    print("\nfiles:")
    for _name, _uris in sorted(
        data_fetch.dataset_files(_result.slices, _root_uri).items()
    ):
        print(f"  {_name}")
        for _uri in _uris:
            print(f"    {_uri}")
