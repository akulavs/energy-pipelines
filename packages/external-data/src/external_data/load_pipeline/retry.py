"""
Retry timing shared by the load pipeline's two download helpers.

OEDI publishes no rate limit, so a server-sent ``Retry-After`` is the only budget
figure there is to honour; everything else is exponential backoff. Mirrors
``climate_pipeline.nsrdb.bronze._retry_wait``. Its own module because
``oedi_building_stock`` and ``dsg_common`` both need it and neither should import
the other -- they are siblings, not layers.
"""

from __future__ import annotations

import httpx

# Cap Retry-After so a hostile or mistaken header cannot stall a batch. Matches
# the climate pipeline's figure; both are a bound on one wait, not a budget.
MAX_RETRY_AFTER_SECONDS = 120.0


def retry_wait(response: httpx.Response, backoff_seconds: float, attempt: int) -> float:
    """
    Seconds to wait before the next retry: honour a ``Retry-After`` header (in
    seconds) when the server sends one, else exponential backoff.

    The HTTP-date form falls back to backoff rather than being parsed -- S3 and
    CloudFront send delta-seconds, so a date parser would be an untested path;
    the fallback is merely slower, never wrong.
    """
    retry_after = response.headers.get("Retry-After")
    if retry_after is not None:
        try:
            # Clamped at both ends: the cap bounds one wait, and the floor keeps a
            # negative header out of ``time.sleep``, which would raise inside the
            # retry loop instead of backing off.
            return max(0.0, min(float(retry_after), MAX_RETRY_AFTER_SECONDS))
        except ValueError:
            pass  # HTTP-date form -- fall back to exponential backoff
    return backoff_seconds * (2 ** (attempt - 1))
