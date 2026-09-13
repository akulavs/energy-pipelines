"""
Which bronze slices are already on disk.

Answers "do I already have this?" from the manifest -- the durable ledger, not a
TTL-bound cache whose miss weeks later would silently re-fetch. Without it every
re-run re-pays every request and a run killed partway resumes from nothing.

Simpler than the climate pipeline's ``point_manifest`` by one dimension. There, a
stored write covers a request only if its date range *contains* it, so coverage is a
comparison. Here every source publishes exactly one modelled year and no request
narrows it, so a key either exists or it does not. That test is ``params_json``
equality -- exactly what :func:`common.storage.manifest.resolve_manifest` applies, so
a hit here is a hit there by construction.

**One scan per dataset, filtered in memory**, never resolve-per-key: a run plans one
key per PUMA, and resolving them one at a time would be a manifest scan apiece to
answer a single question.

In the package because a manifest read is domain work; the flows decide what to do
about the answer.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Sequence

import pydantic

from common.storage.manifest import ManifestRow, scan_manifest

logger = logging.getLogger(__name__)


def existing_params(
    dataset_name: str,
    root_uri: str,
    *,
    as_of: dt.datetime | None = None,
) -> dict[str, ManifestRow]:
    """
    Every ``params_json`` already written for *dataset_name*, mapped to its write.

    ``scan_manifest`` already returns the latest row per unique params, so a key
    written several times resolves here to the same version a reader would get.
    """
    return {
        row.params_json: row
        for row in scan_manifest(
            dataset_name=dataset_name, root_uri=root_uri, as_of=as_of
        )
    }


def split_by_coverage[P: pydantic.BaseModel](
    dataset_name: str,
    params: Sequence[P],
    root_uri: str,
    *,
    force_refresh: bool = False,
    as_of: dt.datetime | None = None,
) -> tuple[list[P], list[ManifestRow]]:
    """
    Split planned writes into those still needed and the writes covering the rest.

    One scan for the whole list, whatever the caller does next. ``force_refresh``
    re-fetches everything, which is the escape hatch for a re-published release
    landing under an unchanged key.
    """
    if force_refresh:
        return list(params), []

    have = existing_params(dataset_name, root_uri, as_of=as_of)
    to_write: list[P] = []
    covered: list[ManifestRow] = []
    for entry in params:
        row = have.get(entry.model_dump_json())
        if row is None:
            to_write.append(entry)
        else:
            covered.append(row)
    if covered:
        logger.info(
            "%s: %d of %d slice(s) already written",
            dataset_name,
            len(covered),
            len(params),
        )
    return to_write, covered


def covered_write(
    dataset_name: str,
    params: pydantic.BaseModel,
    root_uri: str,
    *,
    force_refresh: bool = False,
    as_of: dt.datetime | None = None,
) -> ManifestRow | None:
    """
    The existing write for exactly *params*, or ``None`` if it still has to be
    fetched. The single-key form of :func:`split_by_coverage`.
    """
    _, covered = split_by_coverage(
        dataset_name, [params], root_uri, force_refresh=force_refresh, as_of=as_of
    )
    return covered[0] if covered else None
