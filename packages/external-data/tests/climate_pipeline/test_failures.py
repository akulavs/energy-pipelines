"""
Tests for permanent-vs-transient fetch failure classification.

Getting this backwards is expensive in both directions, which is why the default
matters: a transient failure wrongly called permanent silently drops a point from
the run, while a permanent one wrongly called transient only wastes a couple of
retries. The tests below pin that asymmetry -- anything unrecognised stays
transient.
"""

from __future__ import annotations

import pytest

import common.exceptions
from external_data.climate_pipeline import failures


def test_permanent_error_is_a_pipeline_value_error() -> None:
    # Subclassing keeps existing `except PipelineValueError` handlers working;
    # the distinct type is only what the retry policy keys on.
    exc = failures.PermanentFetchError("nope")
    assert isinstance(exc, common.exceptions.PipelineValueError)
    assert "nope" in str(exc)


def test_is_permanent_only_for_the_marker_type() -> None:
    assert failures.is_permanent(failures.PermanentFetchError("x")) is True
    assert failures.is_permanent(common.exceptions.PipelineValueError("x")) is False
    assert failures.is_permanent(RuntimeError("x")) is False
    assert failures.is_permanent(None) is False


def test_a_plain_value_error_is_not_permanent() -> None:
    # A live run raised a bare PipelineValueError for a request that succeeded
    # unchanged seconds later, so the base type must stay retryable.
    assert (
        failures.is_permanent(
            common.exceptions.PipelineValueError(
                "no NSRDB rows fell inside [2023-01-01, 2023-01-27] after trimming 1"
            )
        )
        is False
    )


# --------------------------------------------------------------------------- #
# CDS classification (message-based: cdsapi raises an untyped Exception)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 410, 422])
def test_cds_client_errors_are_permanent(status: int) -> None:
    # cdsapi retries 408/429/5xx itself, so a status that reaches us is one it
    # already judged non-retriable.
    exc = Exception(f"HTTP error: [{status} Client Error]")
    assert isinstance(failures.as_permanent_cds(exc), failures.PermanentFetchError)


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_cds_retriable_statuses_stay_transient(status: int) -> None:
    assert failures.as_permanent_cds(Exception(f"HTTP error: [{status} Boom]")) is None


@pytest.mark.parametrize(
    "message",
    [
        "Request has not produced any data. Please check your selection.",
        "No data is available within your requested subset.",
        "Invalid request. The date is out of range.",
        "Required licence not accepted for this dataset.",
    ],
)
def test_cds_job_failures_that_can_never_succeed_are_permanent(message: str) -> None:
    assert isinstance(
        failures.as_permanent_cds(Exception(message)), failures.PermanentFetchError
    )


@pytest.mark.parametrize(
    "message",
    [
        "Connection reset by peer",
        "The job failed with an internal error.",
        "Could not connect",
        "",
    ],
)
def test_unrecognised_cds_messages_stay_transient(message: str) -> None:
    # The safe default: retrying costs seconds, dropping a point costs data.
    assert failures.as_permanent_cds(Exception(message)) is None


def test_cds_classification_keeps_the_original_text() -> None:
    permanent = failures.as_permanent_cds(
        Exception("HTTP error: [400 Bad Request] bad coordinates")
    )
    assert permanent is not None
    assert "bad coordinates" in str(permanent)


# --------------------------------------------------------------------------- #
# NSRDB 4xx: which are really permanent
# --------------------------------------------------------------------------- #

# Bodies measured against the live API.
_OCEAN = '{"errors":["No data available at the provided location","Data processing failure."]}'
_BAD_YEAR = '{"errors":["Invalid value(s)"]}'
_BACKEND = '{"errors":["Data processing failure."]}'


def test_no_data_at_location_is_permanent() -> None:
    assert failures.as_permanent_nsrdb(400, _OCEAN) is not None


def test_invalid_values_is_permanent() -> None:
    assert failures.as_permanent_nsrdb(400, _BAD_YEAR) is not None


def test_a_bare_processing_failure_is_transient() -> None:
    # The bug this fixes. "Data processing failure." is NSRDB reporting its own
    # backend, and the identical request succeeds on retry -- but it also appears
    # alongside the genuinely permanent ocean response, so it cannot be the
    # discriminator in either direction.
    assert failures.as_permanent_nsrdb(400, _BACKEND) is None


def test_the_shared_phrase_does_not_make_a_real_rejection_transient() -> None:
    # Guards the other half: the ocean body contains the transient phrase too.
    assert "Data processing failure." in _OCEAN
    assert failures.as_permanent_nsrdb(400, _OCEAN) is not None


@pytest.mark.parametrize("status", [401, 403, 404])
def test_credential_and_endpoint_failures_are_permanent(status: int) -> None:
    # A retry re-sends the same rejected request.
    assert failures.as_permanent_nsrdb(status, "API_KEY_INVALID") is not None


def test_an_unrecognised_body_stays_transient() -> None:
    # Transient by default, as everywhere else in this module: a needlessly
    # retried transient costs seconds, a wrongly permanent one loses a point.
    assert failures.as_permanent_nsrdb(400, '{"errors":["Something new"]}') is None
