"""
Dataset catalog: the registry mapping a ``dataset_name`` to the schema and params
model that define it.

The storage layer is schema-driven -- :func:`common.storage.columnar.write_dataset`
validates against a :class:`~common.frames.BaseDataFrameSchema`, and the params
model is the manifest lookup key -- but nothing else records *what* a dataset name
is. This module does.

``common`` owns the registry and the base types an entry refers to, and never
imports a domain package. Each domain package depends on ``common`` and registers
its own datasets by calling :func:`register`; whoever composes an application
imports those packages and triggers their registration at startup.
"""

from __future__ import annotations

import dataclasses

import pydantic

from common.frames import BaseDataFrameSchema


@dataclasses.dataclass(frozen=True, kw_only=True)
class DatasetType:
    """A DataFrame-backed dataset (see :mod:`common.storage.columnar`).

    Attributes:
        name:         Stable ``dataset_name`` used in the manifest.
        schema:       The :class:`~common.frames.BaseDataFrameSchema` writes and
                      reads are validated against.
        params_model: Frozen Pydantic model used as the manifest lookup key.
        package:      Owning package -- the answer to "where does this come from".
        description:  Human-readable summary.
    """

    name: str
    schema: type[BaseDataFrameSchema]
    params_model: type[pydantic.BaseModel]
    package: str = ""
    description: str = ""


_CATALOG: dict[str, DatasetType] = {}


def register(entry: DatasetType) -> None:
    """Add *entry* to the catalog.

    Raises:
        ValueError: If the name is already registered. A duplicate almost always
            means the same package was wired in twice; failing loudly surfaces it.
    """
    if entry.name in _CATALOG:
        msg = f"dataset type already registered: {entry.name!r}"
        raise ValueError(msg)
    _CATALOG[entry.name] = entry


def get(name: str) -> DatasetType:
    """The registered entry for *name*.

    Raises:
        KeyError: If nothing is registered under *name*.
    """
    if name not in _CATALOG:
        msg = f"no dataset type registered for {name!r}"
        raise KeyError(msg)
    return _CATALOG[name]


def get_as(name: str, kind: type[DatasetType]) -> DatasetType:
    """The entry for *name*, required to be of *kind*.

    Raises:
        KeyError: If nothing is registered under *name*, or it is not a *kind*.
    """
    entry = get(name)
    if not isinstance(entry, kind):
        msg = f"{name!r} is not registered as a {kind.__name__}"
        raise KeyError(msg)
    return entry


def names() -> list[str]:
    """All registered names, sorted."""
    return sorted(_CATALOG)


def all_types() -> list[DatasetType]:
    """All registered entries, sorted by name."""
    return [_CATALOG[name] for name in names()]


def reset() -> None:
    """Clear the catalog. For test isolation."""
    _CATALOG.clear()
