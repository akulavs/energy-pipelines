"""
Exception types shared by every package in the repo.
"""

import types
from typing import Any


class PipelineExceptionMixin:
    """
    Common interface for the repo's exceptions. Mix into an `Exception` subclass.
    """

    def __init__(self, message: str, **kwargs: Any) -> None:
        self._message = message
        super().__init__(message, **kwargs)  # ty:ignore[too-many-positional-arguments]

    @property
    def message(self) -> str:
        return self._message

    @property
    def traceback(self) -> types.TracebackType | None:
        return getattr(self, "__traceback__", None)

    @property
    def notes(self) -> list[str]:
        return getattr(self, "__notes__", [])

    def __str__(self) -> str:
        return f"{type(self).__name__}: {self.message}"


class PipelineError(PipelineExceptionMixin, Exception):
    """
    The most generic exception. Raise a more specific one when possible.
    """


class PipelineTypeError(PipelineExceptionMixin, TypeError): ...


class PipelineValueError(PipelineExceptionMixin, ValueError): ...
