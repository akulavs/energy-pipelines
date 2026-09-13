"""Tests for common.storage.catalog — the dataset type registry."""

from __future__ import annotations

import collections.abc

import patito as pt
import polars as pl
import pydantic
import pytest

from common.frames import BaseDataFrameSchema
from common.storage import catalog


class _ExampleSchema(BaseDataFrameSchema):
    value: int = pt.Field(dtype=pl.Int32)


class _ExampleParams(pydantic.BaseModel, frozen=True):
    key: str


def _dataset_type(name: str = "example") -> catalog.DatasetType:
    return catalog.DatasetType(
        name=name,
        schema=_ExampleSchema,
        params_model=_ExampleParams,
        description="example dataset",
    )


@pytest.fixture(autouse=True)
def _clean_catalog() -> collections.abc.Generator[None]:
    """Each test starts and ends with an empty catalog."""
    catalog.reset()
    yield
    catalog.reset()


class TestRegister:
    def test_register_then_get(self) -> None:
        dt = _dataset_type()
        catalog.register(dt)
        assert catalog.get("example") is dt

    def test_duplicate_name_raises(self) -> None:
        catalog.register(_dataset_type())
        with pytest.raises(ValueError, match="already registered"):
            catalog.register(_dataset_type())

    def test_distinct_names_coexist(self) -> None:
        catalog.register(_dataset_type("a"))
        catalog.register(_dataset_type("b"))
        assert catalog.names() == ["a", "b"]


class TestGet:
    def test_unknown_name_raises(self) -> None:
        with pytest.raises(KeyError, match="no dataset type registered"):
            catalog.get("missing")


class TestListing:
    def test_names_sorted(self) -> None:
        catalog.register(_dataset_type("zebra"))
        catalog.register(_dataset_type("apple"))
        assert catalog.names() == ["apple", "zebra"]

    def test_all_types_sorted_by_name(self) -> None:
        catalog.register(_dataset_type("zebra"))
        catalog.register(_dataset_type("apple"))
        assert [dt.name for dt in catalog.all_types()] == ["apple", "zebra"]

    def test_empty_by_default(self) -> None:
        assert catalog.names() == []
        assert catalog.all_types() == []


class TestGetAs:
    def test_returns_matching_kind(self) -> None:
        catalog.register(_dataset_type("ds"))
        assert catalog.get_as("ds", catalog.DatasetType).schema is _ExampleSchema

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(KeyError, match="no dataset type registered"):
            catalog.get_as("missing", catalog.DatasetType)


class TestReset:
    def test_reset_clears(self) -> None:
        catalog.register(_dataset_type())
        catalog.reset()
        assert catalog.names() == []
