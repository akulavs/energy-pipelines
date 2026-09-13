"""
Defaults for `StrEnum` and `pydantic.BaseModel` subclasses.
"""

import collections.abc
import enum
from typing import Any, Self

import pydantic

import common.exceptions


class CaseInsensitiveStrEnum(enum.StrEnum):
    """
    A `StrEnum` that also resolves members from `str` values by matching against both
    member names and values, case insensitively.
    """

    @classmethod
    def _missing_(cls, value: Any) -> Self:
        normalized = str(value).strip().lower()

        for i in cls:
            if normalized == i.value or normalized.upper() == i.name:
                return i

        msg = f"invalid {cls.__name__} value: {value}"
        raise common.exceptions.PipelineValueError(msg)


class StrictModel(
    pydantic.BaseModel, validate_default=True, validation_error_cause=True
):
    """
    `pydantic.BaseModel` with defaults validated and validation causes attached.
    """

    def model_copy(
        self,
        *,
        update: collections.abc.Mapping[str, Any] | None = None,
        deep: bool = True,
    ) -> Self:
        """
        Copy, or initialize a new instance when `update`s are passed.

        A copy with updates is re-validated from scratch so every validator sees the
        new values, which the upstream `model_copy` does not do.
        """
        if updates := dict(update or {}):
            return self.model_validate(self.model_dump(mode="python") | updates)
        return super().model_copy(deep=deep)


class FrozenModel(StrictModel):
    """
    A nominally immutable `StrictModel`.
    """

    model_config = pydantic.ConfigDict(frozen=True)
