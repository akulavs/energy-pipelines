"""
Tests for the load pipeline's shared retry timing.

OEDI publishes no rate limit, so ``Retry-After`` is the only budget figure the
server ever volunteers. These cover both halves of honouring it: the arithmetic,
and that the two download helpers actually wait the header's value.
"""

from __future__ import annotations

import time

import httpx
import pytest
import respx

from external_data.load_pipeline import dsg_common, retry
from external_data.load_pipeline import oedi_building_stock as oedi


def _response(headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(429, headers=headers or {})


# --------------------------------------------------------------------------- #
# retry_wait arithmetic
# --------------------------------------------------------------------------- #


def test_honours_a_delta_seconds_header() -> None:
    assert retry.retry_wait(_response({"Retry-After": "7"}), 2.0, 1) == 7.0


def test_the_header_beats_our_own_backoff_in_both_directions() -> None:
    """The server's figure wins whether it is longer or shorter than the guess."""
    longer = retry.retry_wait(_response({"Retry-After": "30"}), 2.0, 1)
    shorter = retry.retry_wait(_response({"Retry-After": "0.5"}), 8.0, 3)
    assert longer == 30.0
    assert shorter == 0.5


def test_a_huge_header_is_capped() -> None:
    """A hostile or mistaken header must not stall a whole batch."""
    wait = retry.retry_wait(_response({"Retry-After": "86400"}), 2.0, 1)
    assert wait == retry.MAX_RETRY_AFTER_SECONDS


def test_a_negative_header_never_reaches_sleep() -> None:
    """
    ``time.sleep`` raises on a negative argument, so an unclamped negative header
    would kill a throttled request with a ValueError instead of backing off --
    inside the retry loop, where the error would look unrelated to the 429 that
    caused it. The floor is the same reasoning as the cap.
    """
    wait = retry.retry_wait(_response({"Retry-After": "-5"}), 2.0, 1)
    assert wait == 0.0
    time.sleep(wait)  # the call the clamp exists to protect


def test_no_header_falls_back_to_exponential_backoff() -> None:
    waits = [retry.retry_wait(_response(), 2.0, attempt) for attempt in (1, 2, 3)]
    assert waits == [2.0, 4.0, 8.0]


def test_an_http_date_header_falls_back_to_backoff() -> None:
    """
    The date form is not parsed -- S3 and CloudFront send delta-seconds, so a date
    parser would be an untested path. Falling back is slower, never wrong.
    """
    wait = retry.retry_wait(
        _response({"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}), 2.0, 2
    )
    assert wait == 4.0


# --------------------------------------------------------------------------- #
# Both download helpers use it
# --------------------------------------------------------------------------- #

_URL = "https://oedi-data-lake.s3.amazonaws.com/some/object.parquet"


@pytest.mark.parametrize(
    "download",
    [oedi.download_object, dsg_common.download_dsg],
    ids=["download_object", "download_dsg"],
)
@respx.mock
def test_download_helpers_wait_the_header_value(
    download, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A 429 carrying Retry-After must sleep for *that* value, not the local backoff.

    Both helpers hit the same bucket, so both have to honour it -- a fix in one
    that missed the other would leave the largest objects the pipeline pulls
    (the ``.dsg`` files) still ignoring back-pressure.
    """
    slept: list[float] = []
    module = dsg_common if download is dsg_common.download_dsg else oedi
    monkeypatch.setattr(module.time, "sleep", slept.append)
    respx.get(_URL).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "11"}, text="SlowDown"),
            httpx.Response(200, content=b"payload"),
        ]
    )

    assert download(_URL, max_retries=3, backoff_seconds=2.0) == b"payload"
    # 11 from the header, not 2.0 from backoff_seconds
    assert slept == [11.0]
