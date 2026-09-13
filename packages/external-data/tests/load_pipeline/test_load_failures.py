"""
Tests for the load pipeline's failure classification.

OEDI is a plain public object store, so status alone classifies -- unlike the
climate pipeline, whose providers report real rejections and their own backend
failures with the same status and have to be matched on message phrases.
"""

from __future__ import annotations

import pytest

from external_data.load_pipeline import failures

_URL = "https://oedi-data-lake.s3.amazonaws.com/some/1234-0.parquet"


@pytest.mark.parametrize("status", [400, 401, 403, 404, 410, 451])
def test_a_non_429_4xx_is_permanent(status: int) -> None:
    """A key that is not readable does not become readable on a retry."""
    permanent = failures.as_permanent_oedi(status, _URL, "NoSuchKey")
    assert isinstance(permanent, failures.PermanentFetchError)
    assert failures.is_permanent(permanent)
    assert str(status) in str(permanent)


def test_429_stays_transient() -> None:
    """Throttling is the one 4xx that means "later", not "never"."""
    assert failures.as_permanent_oedi(429, _URL, "SlowDown") is None


@pytest.mark.parametrize("status", [200, 301, 500, 503])
def test_non_4xx_stays_transient(status: int) -> None:
    assert failures.as_permanent_oedi(status, _URL, "") is None


def test_an_unrelated_exception_is_not_permanent() -> None:
    """Transient is the default: only the raise site can classify."""
    assert failures.is_permanent(ValueError("something else")) is False
    assert failures.is_permanent(None) is False


def test_a_record_carries_its_ids_as_fields() -> None:
    """
    Not only inside a display string.

    The climate pipeline learned this the hard way: records written before the
    coordinates had their own fields are unmatchable later, so they suppress
    nothing and the ledger's formatting becomes load-bearing.
    """
    entry = failures.record(1234, "G11000101", "resstock", permanent=True, error="404")
    assert entry["bldg_id"] == "1234"
    assert entry["puma_gisjoin"] == "G11000101"
    assert entry["source"] == "resstock"
    assert entry["permanent"] == "True"
