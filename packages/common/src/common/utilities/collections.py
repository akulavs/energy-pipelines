"""
Functions that operate on collections.
"""

import collections.abc

import more_itertools


def get_unique[T: collections.abc.Hashable](
    items: collections.abc.Iterable[T],
) -> tuple[T, ...]:
    """
    Return a tuple of unique elements, preserving order.
    """
    return tuple(dict.fromkeys(items))


def flatten(*items: collections.abc.Iterable) -> tuple:
    """
    Flatten `items`. `str` and `bytes` values are _not_ considered iterable, e.g. `flatten("ab") -> ("ab",)`.
    """
    return tuple(more_itertools.collapse(items))
