"""
The record of points a provider permanently could not serve.

A scattered request always contains some. Change 4 of the design note records
them so a run can finish over the gap; this module is the other half -- reading
them back, so a later run stops re-asking a provider for an answer it has already
given, and so silver can tell an *expected* gap from bronze that was simply never
ingested.

Lives in the package rather than in the flow because the record is a **persisted
format**: the writer and every later reader have to agree on it, which makes it a
domain contract, not orchestration. Nothing here imports Prefect; the flow decides
*when* to record and hold back, this decides *what* that means.

Records live in each run's flow manifest at ``{root_uri}/_flows/{flow_id}.json``
under ``metadata.failed_points``.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterable, Mapping, Sequence

from common.storage.flow_manifest import query_flow_manifests
from common.storage.manifest import ManifestRow

from external_data.climate_pipeline import point_manifest

# Flows whose manifests carry failure records. The two bronze flows write a flat
# list for their one source; the pipeline writes both sources keyed by name.
_FLAT_FLOW = {"era5": "ingest_era5", "nsrdb": "ingest_nsrdb"}
_PIPELINE_FLOW = "run_climate_pipeline"


def record(
    point: tuple[float, float], *, permanent: bool, error: str
) -> dict[str, str]:
    """
    One recorded failure: the point in readable *and* parseable form.

    Coordinates get their own fields so :func:`known_points` can match them later
    without the display string's format becoming load-bearing.
    """
    return {
        "point": f"({point[0]}, {point[1]})",
        "latitude": str(point[0]),
        "longitude": str(point[1]),
        "permanent": str(permanent),
        "error": error,
    }


def known_points(
    root_uri: str, source: str, grain: point_manifest.Grain
) -> dict[point_manifest.Point, str]:
    """
    Points earlier runs recorded as permanently unserviceable for *source*.

    Without this every run re-asks each provider for points it has already
    refused, and a later silver run cannot tell an expected gap from bronze never
    ingested.

    Matched at *grain*, so a coordinate metres from a recorded dead one is still
    recognised. Records written before the coordinates were stored separately are
    unmatchable and simply suppress nothing.
    """
    dead: dict[point_manifest.Point, str] = {}
    for flow_name in (_FLAT_FLOW[source], _PIPELINE_FLOW):
        for run in query_flow_manifests(root_uri=root_uri, flow_name=flow_name):
            recorded = (run.metadata or {}).get("failed_points")
            if isinstance(recorded, dict):
                recorded = recorded.get(source, [])
            for entry in recorded or []:
                if not isinstance(entry, dict) or entry.get("permanent") != "True":
                    continue
                try:
                    key = grain((float(entry["latitude"]), float(entry["longitude"])))
                except KeyError, TypeError, ValueError:
                    continue
                dead[key] = entry.get("error", "recorded permanently unavailable")
    return dead


def hold_back(
    to_fetch: Sequence[tuple[float, float]],
    root_uri: str,
    source: str,
    grain: point_manifest.Grain,
    *,
    force_refresh: bool,
) -> tuple[list[tuple[float, float]], list[dict[str, str]]]:
    """
    Split off points already recorded as permanently unserviceable.

    Held points come back as failure records rather than being dropped, so one
    path does both jobs: the run stops spending requests, and the gap stays
    recorded for silver. ``force_refresh`` skips the list -- the way back for a
    point judged dead in error.
    """
    if force_refresh:
        return list(to_fetch), []
    dead = known_points(root_uri, source, grain)
    if not dead:
        return list(to_fetch), []
    keep: list[tuple[float, float]] = []
    held: list[dict[str, str]] = []
    for point in to_fetch:
        reason = dead.get(grain(point))
        if reason is None:
            keep.append(point)
        else:
            held.append(
                record(
                    point,
                    permanent=True,
                    error=f"not re-attempted, recorded permanently unavailable: {reason}",
                )
            )
    return keep, held


def short_writes(
    rows: Sequence[ManifestRow], requested_end: dt.date
) -> list[dict[str, str]]:
    """
    Writes covering less than was asked for, reported like a failed point.

    A narrowed write succeeds *and* leaves a gap; listing it beside the outright
    failures keeps that recorded rather than only logged, and lets the caller
    build over it. Permanent by construction -- a transient year failure fails the
    point instead of narrowing it.
    """
    short: list[dict[str, str]] = []
    for row in rows:
        # Unguarded, unlike the scanning readers: these are writes this run just
        # made, so the key is current by construction. A failure here is a bug in
        # the writer, not a legacy row to skip past.
        point, _start, end = point_manifest.read_point_key(json.loads(row.params_json))
        if end >= requested_end:
            continue
        short.append(
            record(
                point,
                permanent=True,
                error=(
                    f"covers only through {end}, not {requested_end}: "
                    "later year(s) permanently unavailable"
                ),
            )
        )
    return short


def unexplained(
    root_uri: str,
    missing: Mapping[str, tuple[Iterable[point_manifest.Point], point_manifest.Grain]],
) -> list[str]:
    """
    Missing locations no earlier run explained as permanently unserviceable.

    A gap is expected when a run recorded the provider refusing that point,
    unexpected when bronze was never run for it. Only the second should stop a
    build -- otherwise the caller waives the check for *every* gap, including the
    ones worth failing on.

    Each source names itself and supplies its own grain, since a record only
    matches at the grain its fetch wrote. That keeps this module clear of the
    provider packages; the sources it can answer for are still the ones
    :data:`_FLAT_FLOW` knows.
    """
    unrecorded: list[str] = []
    for source, (points, grain) in missing.items():
        # One scan per source, not per point.
        dead = known_points(root_uri, source, grain)
        unrecorded.extend(f"{source} {p}" for p in sorted(points) if p not in dead)
    return unrecorded


def describe(missing: Mapping[str, Iterable[point_manifest.Point]]) -> list[str]:
    """Requested locations with no bronze, labelled by which source lacks them."""
    return [
        f"{source} {p}" for source, points in missing.items() for p in sorted(points)
    ]
