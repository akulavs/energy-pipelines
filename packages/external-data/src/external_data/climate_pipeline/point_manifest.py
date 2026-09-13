"""
Per-point manifest lookups shared by the climate bronze modules and silver.

Every write is **one manifest entry per point**, which is what makes the pipeline
incremental, resumable, and safe to fetch concurrently -- each point owns its own
immutable write, so nothing contends. The manifest is the spatial index: no
physical partitions, no directory layout, just a scan filtered in memory.

This module holds what all three sources share -- scanning, matching a point, and
deciding whether a write *covers* a requested range.

**Scaling note.** One sidecar per point means the *manifest* accumulates small
files, and every lookup reads all of them. A non-issue up to hundreds of points;
in the thousands, the fix is to compact the sidecars into one Parquet file the
read path prefers. Chunked writes are the fallback if compaction is not enough --
don't add them pre-emptively.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from collections.abc import Callable, Iterable, Sequence

from common.storage.manifest import ManifestRow, scan_manifest

logger = logging.getLogger(__name__)

# Compare coordinates at fixed precision, so a re-request of "the same" point
# matches the entry it already wrote.
POINT_DECIMALS = 6

Point = tuple[float, float]
# Pulls (point, start_date, end_date) out of a per-point key's params JSON.
KeyReader = Callable[[dict], tuple[Point, dt.date, dt.date]]
# The identity a *fetch* actually has: exact coordinates for NSRDB, the grid node
# for ERA5 (CDS snaps to it, so two points in a node share one download). A lookup
# must use the grain the write used, or the entry is invisible and refetched.
Grain = Callable[[Sequence[float]], Point]


def normalise(point: Sequence[float]) -> Point:
    """A point rounded to the precision manifest lookups compare at."""
    return (
        round(float(point[0]), POINT_DECIMALS),
        round(float(point[1]), POINT_DECIMALS),
    )


def read_point_key(params: dict) -> tuple[Point, dt.date, dt.date]:
    """
    Pull ``(point, start, end)`` out of a per-point key's params JSON.

    One reader for every source; the key is single-point everywhere now.

    Legacy shapes are still accepted -- ``points`` as a one-element list, at the
    top level for ERA5/silver and nested under ``request`` for NSRDB -- so
    renaming the key did not orphan writes already on disk. That branch can go
    once no such entries remain.

    A legacy key holding *several* points is rejected rather than read as its
    first: claiming a batch entry for ``points[0]`` would report coverage for one
    location and lose the rest. Raising is what makes :func:`existing_points`
    skip it.
    """
    if "request" in params:  # legacy NSRDB: the slice nested under the request
        params = params["request"]
    if "point" in params:
        point = params["point"]
    else:
        points = params["points"]  # legacy: a collection of exactly one
        if len(points) != 1:
            msg = f"not a per-point key: {len(points)} points"
            raise ValueError(msg)
        point = points[0]
    return (
        normalise(point),
        dt.date.fromisoformat(params["start_date"]),
        dt.date.fromisoformat(params["end_date"]),
    )


def existing_points(
    dataset_name: str,
    root_uri: str,
    key_reader: KeyReader,
    *,
    grain: Grain = normalise,
    as_of: dt.datetime | None = None,
) -> dict[Point, tuple[ManifestRow, dt.date, dt.date]]:
    """
    Every point already written for *dataset_name*, with the range it covers.

    Keyed at *grain*, so callers get back the identity they look up by. Two writes
    collapsing to one key resolve to the most recent.

    One scan for the whole dataset, not a lookup per point -- the per-point loop
    is what gets slow at hundreds of points, not the parquet reads.
    """
    found: dict[Point, tuple[ManifestRow, dt.date, dt.date]] = {}
    skipped = 0
    for row in scan_manifest(dataset_name=dataset_name, root_uri=root_uri, as_of=as_of):
        try:
            point, start, end = key_reader(json.loads(row.params_json))
        except KeyError, IndexError, ValueError, TypeError:
            # A batch-era entry, or a key shape this reader does not understand.
            # Skipping is right: it cannot answer "do I have this point?".
            skipped += 1
            continue
        key = grain(point)
        current = found.get(key)
        # Compare write times rather than letting the last row scanned win: scan
        # order is not part of the contract, and relying on it previously kept the
        # *oldest* entry. A point has several entries whenever its writes covered
        # different ranges, and picking the older, narrower one hides bronze that
        # exists -- the point refetches forever and silver reads the short file.
        if current is None or row.write_time > current[0].write_time:
            found[key] = (row, start, end)
    if skipped:
        # Explains a surprising refetch: pre-per-point writes are invisible here,
        # so a dataset written before this layout looks empty.
        logger.info(
            "%s: %d manifest row(s) are not per-point and were ignored",
            dataset_name,
            skipped,
        )
    return found


def covers(
    covered: tuple[ManifestRow, dt.date, dt.date] | None,
    start: dt.date,
    end: dt.date,
) -> bool:
    """Does an existing write span the whole requested range?"""
    if covered is None:
        return False
    _, have_start, have_end = covered
    return have_start <= start and have_end >= end


def covered_rows(
    points: Iterable[Point],
    dataset_name: str,
    root_uri: str,
    key_reader: KeyReader,
    start: dt.date,
    end: dt.date,
    *,
    grain: Grain = normalise,
    as_of: dt.datetime | None = None,
    match: Callable[[ManifestRow], bool] | None = None,
) -> tuple[dict[Point, ManifestRow], set[Point]]:
    """
    The write covering each requested point, plus the requested points with none.

    One scan for the whole request, whatever the caller does next -- a reader that
    then loops points still resolves them all in one pass.

    Returns the whole :class:`ManifestRow`, not just ``data_uri``: callers need
    the location to read *and* the ``write_id`` to record as lineage. Returning
    only the URI previously forced a second copy of this rule for the lineage path.

    ``match`` filters on identity kept outside the point and range (NSRDB's
    interval). Callers own their "missing" wording, which differs per dataset.
    """
    have = existing_points(dataset_name, root_uri, key_reader, grain=grain, as_of=as_of)
    wanted = {grain(p) for p in points}
    found: dict[Point, ManifestRow] = {}
    for point, entry in have.items():
        if point not in wanted or not covers(entry, start, end):
            continue
        if match is not None and not match(entry[0]):
            continue
        found[point] = entry[0]
    return found, wanted - found.keys()


def interval_match(interval: int | None) -> Callable[[ManifestRow], bool] | None:
    """
    A :func:`covered_rows` filter for NSRDB's ``interval``.

    30- and 60-minute fetches of one point are different datasets, so a lookup
    ignoring the interval resolves the wrong slice.
    """
    if interval is None:
        return None

    def _matches(row: ManifestRow) -> bool:
        return json.loads(row.params_json).get("interval") == interval

    return _matches


def split_by_coverage(
    points: Iterable[Point],
    dataset_name: str,
    root_uri: str,
    key_reader: KeyReader,
    start: dt.date,
    end: dt.date,
    *,
    grain: Grain = normalise,
    force_refresh: bool = False,
    as_of: dt.datetime | None = None,
    match: Callable[[ManifestRow], bool] | None = None,
) -> tuple[list[Point], list[ManifestRow]]:
    """
    Split requested points into those needing a fetch and those already covered.

    The fetch list holds the *original* coordinates: the grain decides identity,
    but the fetch wants the caller's point. Decided against the durable manifest,
    never a cache -- an entry written weeks ago is still authoritative, where a
    TTL-bound cache would silently expire and refetch.

    ``match`` filters on identity kept outside the point and range, exactly as on
    :func:`covered_rows`. Planning without it would let a write at one NSRDB
    interval read as coverage for a request at another.
    """
    wanted = [(grain(p), normalise(p)) for p in points]
    if force_refresh:
        return [point for _, point in wanted], []

    have = existing_points(dataset_name, root_uri, key_reader, grain=grain, as_of=as_of)
    to_fetch: list[Point] = []
    reusable: list[ManifestRow] = []
    for key, point in wanted:
        entry = have.get(key)
        # covers() rejects None, but narrow explicitly so the reuse branch cannot
        # be entered with nothing to reuse.
        if (
            entry is not None
            and covers(entry, start, end)
            and (match is None or match(entry[0]))
        ):
            reusable.append(entry[0])
        else:
            to_fetch.append(point)
    return to_fetch, reusable


def covered_spans(
    rows: Iterable[ManifestRow], key_reader: KeyReader
) -> set[tuple[Point, dt.date, dt.date]]:
    """
    Each covering write's point *and* the range it holds.

    Read from the write's own key, since that is all a planner keeps once it
    decides to reuse. The range distinguishes reuse from a wider earlier write.
    """
    found: set[tuple[Point, dt.date, dt.date]] = set()
    for row in rows:
        try:
            found.add(key_reader(json.loads(row.params_json)))
        except KeyError, IndexError, ValueError, TypeError:
            # An unreadable key names no point; the count still includes it.
            continue
    return found


def describe_coverage(
    covered: Iterable[tuple[Point, dt.date, dt.date]], limit: int = 5
) -> str:
    """
    Render points together with the range each stored write actually covers.

    The range is the informative half: two points can both be "already covered" by
    writes of quite different spans, and a write falling short of the request is
    exactly what makes a point refetch unexpectedly. Truncated like
    :func:`describe_points`.
    """
    ordered = sorted(covered)
    shown = ", ".join(
        f"({lat:g}, {lon:g}) {start}..{end}"
        for (lat, lon), start, end in ordered[:limit]
    )
    if len(ordered) > limit:
        shown += f", and {len(ordered) - limit} more"
    return shown


def describe_points(points: Iterable[Point], limit: int = 5) -> str:
    """
    Render a set of points for a message, truncated so it stays readable.

    Used for the points a message is *about* -- missing, reused, or held back.
    Truncated: the same line serves a 3-point request and a 2,000-point backfill.
    """
    ordered = sorted(points)
    shown = ", ".join(f"({lat:g}, {lon:g})" for lat, lon in ordered[:limit])
    if len(ordered) > limit:
        shown += f", and {len(ordered) - limit} more"
    return shown
