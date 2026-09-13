"""
Pydantic-friendly types/annotations.
"""

import collections.abc
import contextlib
import copy
import copyreg
import datetime
import logging
import re
import types
import zoneinfo
from typing import Annotated, get_args

import annotated_types
import pydantic
import pydantic_core
import whenever

import common.utilities.collections


def _get_logging_level(level: int | str) -> int:
    """
    Get the canonical (int-valued) logging level.
    """
    match level:
        case str():
            with contextlib.suppress(KeyError):
                return logging.getLevelNamesMapping()[level.upper()]
            with contextlib.suppress(ValueError):
                return int(level)
        case _:
            return level

    msg = f"invalid logging level: {level}"
    raise ValueError(msg)


type PortNumber = Annotated[int, annotated_types.Interval(ge=0, le=65535)]

# use ints for logging levels, to avoid ambiguity when parsing env var values like "20"
type LoggingLevel = Annotated[
    int,
    annotated_types.Interval(ge=0, le=50),
    pydantic.BeforeValidator(_get_logging_level),
]

type NonEmptyStr = Annotated[str, pydantic.StringConstraints(min_length=1)]

# opinionated: lowercase, no quotes, no delimiters
type SQLIdentifier = Annotated[
    str,
    pydantic.StringConstraints(
        pattern=re.compile(R"[a-zA-Z_][a-zA-Z0-9_]{0,255}", re.ASCII), to_lower=True
    ),
]

type Instant = Annotated[
    whenever.Instant,
    pydantic.GetPydanticSchema(
        lambda source_type, handler: pydantic_core.core_schema.union_schema(
            [
                pydantic_core.core_schema.is_instance_schema(whenever.Instant),
                pydantic_core.core_schema.chain_schema(
                    [
                        pydantic_core.core_schema.is_instance_schema(datetime.datetime),
                        pydantic_core.core_schema.no_info_plain_validator_function(
                            whenever.Instant
                        ),
                    ]
                ),
                pydantic_core.core_schema.chain_schema(
                    [
                        pydantic_core.core_schema.is_instance_schema(str),
                        pydantic_core.core_schema.no_info_plain_validator_function(
                            whenever.Instant.parse_rfc2822
                        ),
                    ]
                ),
                handler(source_type),
            ]
        )
    ),
]


def _zoned_date_time_from_py_datetime(
    value: datetime.datetime,
) -> whenever.ZonedDateTime:
    if value.tzinfo == datetime.UTC:
        value = value.replace(tzinfo=zoneinfo.ZoneInfo("UTC"))
    return whenever.ZonedDateTime(value)


type ZonedDateTime = Annotated[
    whenever.ZonedDateTime,
    pydantic.GetPydanticSchema(
        lambda source_type, handler: pydantic_core.core_schema.union_schema(
            [
                pydantic_core.core_schema.is_instance_schema(whenever.ZonedDateTime),
                pydantic_core.core_schema.chain_schema(
                    [
                        pydantic_core.core_schema.datetime_schema(
                            tz_constraint="aware"
                        ),
                        pydantic_core.core_schema.no_info_plain_validator_function(
                            _zoned_date_time_from_py_datetime
                        ),
                    ]
                ),
                handler(source_type),
            ]
        )
    ),
]

type TimeDelta = Annotated[
    whenever.TimeDelta,
    pydantic.GetPydanticSchema(
        lambda source_type, handler: pydantic_core.core_schema.union_schema(
            [
                pydantic_core.core_schema.is_instance_schema(whenever.TimeDelta),
                pydantic_core.core_schema.chain_schema(
                    [
                        pydantic_core.core_schema.is_instance_schema(
                            datetime.timedelta
                        ),
                        pydantic_core.core_schema.no_info_plain_validator_function(
                            whenever.TimeDelta
                        ),
                    ]
                ),
                handler(source_type),
            ]
        )
    ),
]

type PositiveTimeDelta = Annotated[
    TimeDelta, annotated_types.Predicate(whenever.TimeDelta().__lt__)
]
type NonNegativeTimeDelta = Annotated[
    TimeDelta, annotated_types.Predicate(whenever.TimeDelta().__le__)
]


# collection types
type UniqueTuple[T: collections.abc.Hashable] = Annotated[
    tuple[T, ...],
    pydantic.FailFast(),
    pydantic.AfterValidator(common.utilities.collections.get_unique),
]


# TODO: is this worth keeping?
class _MappingProxyTypeAnnotation:
    """
    Marker class to make `types.MappingProxyType` pydantic-compatible.

    Not intended for direct use. Use the `Mapping` type alias below instead.

    Note: `pydantic.ValidateAs` can't be used because it doesn't introspect the type parameters
    when specializing the `Mapping` type alias.

    Ref: https://github.com/pydantic/pydantic/issues/6868
    """

    @classmethod
    def __get_pydantic_core_schema__(  # noqa: PLW3201
        cls,
        source_type: type,
        handler: pydantic.GetCoreSchemaHandler,
    ) -> pydantic_core.core_schema.CoreSchema:
        type_args = get_args(source_type)

        key_schema: pydantic_core.core_schema.CoreSchema | None = None
        value_schema: pydantic_core.core_schema.CoreSchema | None = None

        match type_args:
            case (kt, vt):
                key_schema = handler.generate_schema(kt)
                value_schema = handler.generate_schema(vt)
            case (kt,):
                key_schema = handler.generate_schema(kt)

        dict_schema = pydantic_core.core_schema.dict_schema(key_schema, value_schema)

        return pydantic_core.core_schema.no_info_after_validator_function(
            lambda value: types.MappingProxyType(copy.deepcopy(value)),
            schema=dict_schema,
            serialization=pydantic_core.core_schema.wrap_serializer_function_ser_schema(
                lambda value, handler: handler(dict(value)),
            ),
        )


def _reduce_mapping_proxy[KT, VT](
    x: types.MappingProxyType[KT, VT],
) -> tuple[type[types.MappingProxyType], tuple[dict[KT, VT]]]:
    """
    Enable `pickle` and `deepcopy` support.
    """
    return types.MappingProxyType, (dict(x),)


copyreg.pickle(types.MappingProxyType, _reduce_mapping_proxy)


type Mapping[KT: collections.abc.Hashable, VT] = Annotated[
    types.MappingProxyType[KT, VT],
    _MappingProxyTypeAnnotation,
]
