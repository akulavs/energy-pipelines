"""Tests for common.exceptions."""

from __future__ import annotations

import copy
import pickle

import pytest

from common.exceptions import PipelineError, PipelineTypeError, PipelineValueError


def test_message_is_kept_and_rendered() -> None:
    exc = PipelineValueError("bad input")
    assert exc.message == "bad input"
    assert str(exc) == "PipelineValueError: bad input"
    assert exc.args == ("bad input",)


def test_subclasses_the_matching_builtin() -> None:
    assert isinstance(PipelineValueError("x"), ValueError)
    assert isinstance(PipelineTypeError("x"), TypeError)
    assert isinstance(PipelineError("x"), Exception)


def test_survives_pickle_and_copy() -> None:
    # A Prefect process runner or result store would need this; an exception
    # whose args were empty could not be rebuilt.
    exc = PipelineValueError("carried across")
    for again in (pickle.loads(pickle.dumps(exc)), copy.deepcopy(exc)):
        assert isinstance(again, PipelineValueError)
        assert again.message == "carried across"


def test_notes_are_exposed() -> None:
    exc = PipelineError("x")
    exc.add_note("context")
    assert exc.notes == ["context"]
    with pytest.raises(PipelineError, match="x"):
        raise exc
