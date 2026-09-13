"""
Base classes for DataFrame shape definitions.

:class:`BaseDataFrameSchema` is the public base for all DataFrame schemas.
It provides type-safe column name references via a ``Cols`` namespace
auto-generated at class-creation time::

    class MySchema(BaseDataFrameSchema):
        value: str = pt.Field(dtype=pl.Utf8)
        count: int = pt.Field(dtype=pl.Int32)

    df.filter(pl.col(MySchema.Cols.value) == "foo")
    MySchema.validate(df)

:class:`_ColsMixin` is the private implementation detail that generates the
``Cols`` namespace.
"""

from __future__ import annotations

from typing import Any, ClassVar

import patito as pt
import pydantic


# ---------------------------------------------------------------------------
# Cols mixin (private implementation detail)
# ---------------------------------------------------------------------------


class _ColsMixin(pt.Model):
    """
    Mixin that auto-generates a ``Cols`` namespace on every subclass.

    The ``Cols`` namespace holds column-name strings as attributes so that
    downstream code can reference column names without magic strings::

        df.filter(pl.col(Readings.Cols.station_id) == "KDCA")
    """

    # Populated dynamically by __init_subclass__; typed as Any so that Pylance
    # allows attribute access on the generated Cols namespace (e.g. Cols.p_min).
    Cols: ClassVar[Any]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Uses __annotations__ from the full MRO — reliably populated during
        # class construction before Pydantic's metaclass finishes.
        fields: dict[str, str] = {}
        for klass in reversed(cls.__mro__):
            fields.update(
                {
                    k: k
                    for k in getattr(klass, "__annotations__", {})
                    if not k.startswith("_")
                }
            )
        cls.Cols = type("Cols", (), fields)


# ---------------------------------------------------------------------------
# Public base for DataFrame schemas
# ---------------------------------------------------------------------------


class BaseDataFrameSchema(_ColsMixin):
    """
    Base class for all DataFrame schemas.

    Subclasses automatically gain a ``Cols`` namespace and Patito validation.
    Use this for any schema that describes a DataFrame shape — whether the
    DataFrame is persisted to storage or used only in-memory::

        class HourlyWeatherSchema(BaseDataFrameSchema):
            temperature_c: float = pt.Field(dtype=pl.Float64)
            wind_speed_ms: float = pt.Field(dtype=pl.Float64)

        HourlyWeatherSchema.validate(df)
        df.filter(pl.col(HourlyWeatherSchema.Cols.temperature_c) > 30)
    """

    model_config = pydantic.ConfigDict(
        validate_default=True,
        validation_error_cause=True,
    )
