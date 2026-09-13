"""
Function for (de)serializing JSON objects.
"""

from typing import Any

import pydantic_core


__all__ = ("from_json", "to_json")


def to_json(value: Any) -> str:
    """
    Serialize `value` as a JSON string. Use instead of `json.dumps`
    """
    return pydantic_core.to_json(value).decode()


def from_json(value: str | bytes | bytearray) -> Any:
    """
    Deserialize `value` to a Python object. Use instead of `json.loads`
    """
    return pydantic_core.from_json(value)
