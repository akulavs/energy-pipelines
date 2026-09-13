"""
Collapse a caller's PUMA list onto the grains the load sources publish at.

Simpler than the climate side's coordinate consolidation, and differently shaped:
a PUMA GISJOIN is already an exact identifier, so there is no grid to snap to and
nothing is approximate. Two grains still matter:

- **The PUMA** is what ResStock and ComStock publish and what every building-stock
  bronze write is keyed by. Repeats collapse to one.
- **The state** is what dsgrid publishes -- one file per state covering every
  county at once -- and what the industrial silver tables are keyed by. Several
  PUMAs of one state need that work done once, not once each.

``run_load_pipeline`` already collapses to distinct states internally when it
decides which dsgrid tasks to submit. Doing it here as well is what lets a caller
ask the manifests what exists *before* starting a run.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Iterable, Mapping

from external_data.load_pipeline import schema

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ConsolidatedGeographies:
    """
    The PUMAs and states a request actually resolves to.

    ``state_by_puma`` is the half a caller needs to read results back: the
    building-stock datasets are keyed by PUMA, the industrial ones by state, so a
    caller point maps into the industrial tables through its state.
    """

    pumas: tuple[schema.LoadGeography, ...]
    states: tuple[str, ...]
    state_by_puma: Mapping[str, str]

    @property
    def geographies(self) -> schema.LoadGeographies:
        """The request model the flow takes."""
        return schema.LoadGeographies(pumas=self.pumas)


def consolidate(
    codes: Iterable[str] | schema.LoadGeographies,
) -> ConsolidatedGeographies:
    """
    Split a list of PUMA GISJOINs into the PUMAs and the states worth working on.

    Order is preserved and the first occurrence wins, so a caller's list still
    reads back in the order they wrote it. A repeated PUMA is dropped rather than
    fetched twice; a state is listed once however many of its PUMAs were asked for.
    """
    if isinstance(codes, schema.LoadGeographies):
        wanted = list(codes.pumas)
    else:
        # One at a time, not through ``LoadGeographies``: that model rejects a
        # repeated PUMA outright, and de-duplicating is exactly this function's
        # job. Validating each code singly still applies the GISJOIN pattern and
        # derives the state, so nothing is skipped -- only the batch-level
        # uniqueness rule, which is enforced below by construction.
        wanted = [
            code
            if isinstance(code, schema.LoadGeography)
            else schema.LoadGeography(puma_gisjoin=code)
            for code in codes
        ]

    pumas: list[schema.LoadGeography] = []
    seen: set[str] = set()
    state_by_puma: dict[str, str] = {}
    states: list[str] = []
    for geography in wanted:
        state_by_puma[geography.puma_gisjoin] = geography.state
        if geography.puma_gisjoin not in seen:
            seen.add(geography.puma_gisjoin)
            pumas.append(geography)
        if geography.state not in states:
            states.append(geography.state)

    logger.info(
        "consolidated %d requested PUMA(s) -> %d distinct PUMA(s) in %d state(s)",
        len(wanted),
        len(pumas),
        len(states),
    )
    return ConsolidatedGeographies(
        pumas=tuple(pumas), states=tuple(states), state_by_puma=state_by_puma
    )
