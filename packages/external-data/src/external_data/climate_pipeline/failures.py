"""
Telling a permanent fetch failure from a transient one.

At thousands of scattered points some fail *permanently* -- a coordinate at sea,
a rejected request -- and retrying them only burns the provider budget the rest
of the run needs. The two call for opposite handling: retry the transient, record
and skip the permanent.

The distinction is made **at the raise site**, not by inspecting messages: only
the code that saw the response knows. Everything unrecognised is transient, the
safe default -- a needlessly retried transient costs seconds, a wrongly permanent
one loses a point until someone passes ``force_refresh``.
"""

from __future__ import annotations

import re

import common.exceptions

# cdsapi raises a bare ``Exception`` whose text is the only signal, so CDS has to
# be classified by message. Brittle by nature, hence short and certain lists --
# anything unmatched stays transient.
_CDS_HTTP_STATUS = re.compile(r"HTTP error:\s*\[(\d{3})")

# cdsapi retries 408/429/5xx itself, so a status reaching us is one it judged
# non-retriable. These are the ones a retry would re-send unchanged.
_CDS_PERMANENT_STATUSES = frozenset({400, 401, 403, 404, 409, 410, 422})

# Job-level failures CDS reports as "<message>. <reason>." Only phrases meaning
# "this can never succeed as written" belong here.
_CDS_PERMANENT_PHRASES = (
    "not produced any data",
    "no data is available",
    "invalid request",
    "required licence",
    "licences not accepted",
)


# NSRDB reports both real rejections and its own backend failures as 400, so the
# status alone cannot classify. Measured against the live API:
#
#   ocean point / impossible latitude
#       ["No data available at the provided location", "Data processing failure."]
#   year outside coverage
#       ["Invalid value(s)"]
#   a request that succeeded unchanged on retry
#       ["Data processing failure."]
#
# "Data processing failure." therefore appears in *both* classes and cannot be the
# discriminator; what distinguishes a real rejection is the phrase naming the
# location or the parameters. Anything unmatched stays transient.
_NSRDB_PERMANENT_PHRASES = (
    "no data available at the provided location",
    "invalid value(s)",
)

# Credentials and endpoint: a retry re-sends the same rejected request.
_NSRDB_PERMANENT_STATUSES = frozenset({401, 403, 404})


class PermanentFetchError(common.exceptions.PipelineValueError):
    """
    A fetch that cannot succeed by trying again.

    Subclasses ``PipelineValueError`` so existing handlers still catch it; the
    distinct type is what lets the retry policy skip it and the flow record it.
    """


def is_permanent(exc: BaseException | None) -> bool:
    """Should this be recorded and skipped rather than retried?"""
    return isinstance(exc, PermanentFetchError)


def as_permanent_nsrdb(status: int, body: str) -> PermanentFetchError | None:
    """
    Classify an NSRDB 4xx, or ``None`` to leave it transient and retryable.

    Narrow on purpose. Treating every 4xx as permanent means one backend hiccup
    skips the retries, gets dead-lettered, and removes a legitimate point from
    every later run -- recoverable only by someone noticing and passing
    ``force_refresh``.
    """
    if status in _NSRDB_PERMANENT_STATUSES:
        return PermanentFetchError(f"NSRDB rejected the request ({status}): {body}")
    lowered = body.lower()
    if any(phrase in lowered for phrase in _NSRDB_PERMANENT_PHRASES):
        return PermanentFetchError(f"NSRDB cannot serve this request: {body}")
    return None


def as_permanent_cds(exc: BaseException) -> PermanentFetchError | None:
    """
    Reclassify a cdsapi error as permanent, or ``None`` to leave it transient.

    The match is deliberately narrow: an unrecognised message stays transient,
    which is the failure mode that merely wastes time.
    """
    text = str(exc)
    status = _CDS_HTTP_STATUS.search(text)
    if status and int(status.group(1)) in _CDS_PERMANENT_STATUSES:
        return PermanentFetchError(f"CDS rejected the request: {text}")
    lowered = text.lower()
    if any(phrase in lowered for phrase in _CDS_PERMANENT_PHRASES):
        return PermanentFetchError(f"CDS cannot serve this request: {text}")
    return None
