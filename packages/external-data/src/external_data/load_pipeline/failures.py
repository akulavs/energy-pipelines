"""
Telling a permanent fetch failure from a transient one.

A PUMA is hundreds of per-building files and some are genuinely absent -- a release
lists a building whose timeseries was never published. Retrying those spends the
run's budget on an answer the source has already given.

Classified **at the raise site** rather than by sniffing messages: OEDI is a plain
public S3 bucket, so the status says everything. (``climate_pipeline.failures`` has
to match on message phrases, because NSRDB and CDS report real rejections and their
own backend failures under the same status.)

Transient is the default for anything unrecognised: a needlessly retried transient
costs seconds, a wrongly permanent one drops a building until someone passes
``force_refresh``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Mapping, Sequence

import common.exceptions
from common.storage.flow_manifest import query_flow_manifests


class PermanentFetchError(common.exceptions.PipelineValueError):
    """
    A fetch that cannot succeed by trying again.

    Subclasses ``PipelineValueError`` so existing handlers still catch it; the
    distinct type is what lets the retry policy skip it and the flow record it.
    """


def is_permanent(exc: BaseException | None) -> bool:
    """Should this be recorded and skipped rather than retried?"""
    return isinstance(exc, PermanentFetchError)


def as_permanent_oedi(status: int, url: str, body: str) -> PermanentFetchError | None:
    """
    Classify an OEDI 4xx, or ``None`` to leave it transient and retryable.

    Every 4xx except 429 is permanent: against a public object store 403/404 mean the
    key is not readable and 400 means the request is malformed, none of which a retry
    changes. 429 is throttling, handled as transient alongside 5xx.
    """
    if status == 429 or not (400 <= status < 500):
        return None
    return PermanentFetchError(
        f"OEDI cannot serve this request ({status}) for {url}: {body[:200].strip()}"
    )


# --------------------------------------------------------------------------- #
# The record of PUMAs a release cannot serve at all
# --------------------------------------------------------------------------- #

# One level above a dead building: not "this PUMA is missing some files" but "this
# release has no such PUMA". A well-formed GISJOIN that no file answers for -- a
# 2020-vintage code against a 2010-vintage release, a state/PUMA pairing that does
# not exist, or simply a typo in a run form.
#
# Recorded so a run can finish over it rather than being killed by it: one bad code
# in a list of five used to cost the other four their whole ingest, which for the
# building-stock sources is thousands of requests and the better part of an hour.
#
# Records live in each run's flow manifest under ``metadata.rejected_pumas``.

REJECTED_PUMAS_KEY = "rejected_pumas"


def rejected(puma_gisjoin: str, source: str, error: str) -> dict[str, str]:
    """
    One PUMA the release could not serve, in the shape :func:`record` uses.

    ``puma_gisjoin`` and ``source`` are their own fields, like ``bldg_id`` is on a
    building record, so a reader can group by either without the error string's
    formatting becoming load-bearing.

    Always permanent, so unlike :func:`record` it takes no flag: a code the release
    has no file for does not start working on the next attempt. What it deliberately
    has no counterpart to is :func:`known_dead` -- see :func:`describe_rejected`.
    """
    return {
        "puma_gisjoin": puma_gisjoin,
        "source": source,
        "permanent": "True",
        "error": error,
    }


def rejected_pumas(records: Iterable[Mapping[str, str]], source: str) -> set[str]:
    """
    The PUMAs *source* rejected, from this run's records.

    Per source rather than pooled: ResStock and ComStock publish from separate
    releases, so a code one cannot serve is not proof the other cannot. Dropping a
    PUMA from both on one source's word would silently discard data that was there.
    """
    return {
        entry["puma_gisjoin"]
        for entry in records
        if entry.get("source") == source and "puma_gisjoin" in entry
    }


def describe_rejected(records: Sequence[Mapping[str, str]]) -> str:
    """
    The rejected PUMAs as one line for an operator, grouped by source.

    There is no ``known_rejected`` to pair with this -- deliberately, and unlike
    :func:`known_dead`. A building the release refuses is a fact about the release,
    worth remembering so a PUMA can settle; a PUMA it refuses is nearly always a
    mistyped code, and remembering that would mean the corrected run got skipped by
    the record of its own typo. So a rejection is reported and never replayed: the
    only cost of re-checking is the metadata read the run was doing anyway.
    """
    by_source: dict[str, list[str]] = {}
    for entry in records:
        by_source.setdefault(entry.get("source", "?"), []).append(
            entry.get("puma_gisjoin", "?")
        )
    return "; ".join(
        f"{source}: {', '.join(sorted(pumas))}"
        for source, pumas in sorted(by_source.items())
    )


# --------------------------------------------------------------------------- #
# The record of buildings a release cannot serve
# --------------------------------------------------------------------------- #

# Records live in each run's flow manifest under ``metadata.failed_buildings``.
# Reading them back is the half that matters: a record nothing consumes leaves every
# run re-asking the provider for files it has already refused.
_FLOW_NAMES = ("ingest_resstock", "ingest_comstock", "run_load_pipeline")


def record(
    bldg_id: int, puma_gisjoin: str, source: str, *, permanent: bool, error: str
) -> dict[str, str]:
    """
    One recorded failure.

    ``bldg_id`` and ``puma_gisjoin`` are their own fields rather than parts of a
    display string, so :func:`known_dead` can match them without the formatting
    becoming load-bearing.
    """
    return {
        "bldg_id": str(bldg_id),
        "puma_gisjoin": puma_gisjoin,
        "source": source,
        "permanent": str(permanent),
        "error": error,
    }


def known_dead(root_uri: str, source: str, puma_gisjoin: str) -> set[int]:
    """
    Buildings earlier runs recorded as permanently unavailable for this PUMA.

    Excluded before anything is fetched, which is what lets a PUMA settle: without it
    a PUMA holding one unpublished building re-requests that file on every run,
    forever.

    Only ``permanent`` records are honoured -- a transient failure is expected to
    succeed later, and skipping it would turn a blip into a permanent hole.
    """
    dead: set[int] = set()
    for flow_name in _FLOW_NAMES:
        for run in query_flow_manifests(root_uri=root_uri, flow_name=flow_name):
            for entry in (run.metadata or {}).get("failed_buildings", []):
                if (
                    entry.get("permanent") == "True"
                    and entry.get("source") == source
                    and entry.get("puma_gisjoin") == puma_gisjoin
                ):
                    with contextlib.suppress(ValueError, TypeError):
                        dead.add(int(entry["bldg_id"]))
    return dead
