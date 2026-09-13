"""
Collapse a caller's raw coordinates onto the grid the pipeline is keyed by.

**One grain, 0.1 degrees.** ERA5-Land is published on that grid and CDS snaps
every request to it, so a node -- not a coordinate -- is the unit a download is
keyed by. Silver is written per node for the same reason. NSRDB is snapped to the
same grid here even though its own cells are finer, because that is the grain its
data is *consumed* at: ``silver._snap_to_nearest_node`` keeps one NSRDB point per
node and discards the rest, so a finer request buys data the join throws away.

Snapping NSRDB to a *finer* decimal place would not help and would quietly break
lookups. Its served cell centres sit on its own ~4 km grid, offset from round
decimals -- a request at ``(37.3, -122.0)`` comes back as ``(37.29, -122.02)`` --
so rounding a request to two decimals produces a coordinate that matches nothing.
The 0.1 degree node works precisely because it is coarse enough to *contain* the
served cell.

So a request is made at the node, and every write is keyed by one.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Iterable, Mapping, Sequence

from external_data.climate_pipeline import point_manifest
from external_data.climate_pipeline.era5 import bronze as era5_bronze

logger = logging.getLogger(__name__)

Point = point_manifest.Point


@dataclasses.dataclass(frozen=True, kw_only=True)
class ConsolidatedPoints:
    """
    The grid nodes a request resolves to, and where each coordinate landed.

    ``nodes`` is what every source is actually called at. ``node_by_input`` is
    how a caller gets back from what they typed to what came back: the frames'
    coordinate columns are nodes, never the requested coordinate.
    """

    nodes: tuple[Point, ...]
    node_by_input: Mapping[Point, Point]

    @property
    def era5_points(self) -> tuple[Point, ...]:
        """The ERA5 request points -- the nodes, since CDS snaps to them anyway."""
        return self.nodes

    @property
    def nsrdb_points(self) -> tuple[Point, ...]:
        """The NSRDB request points -- also the nodes; see the module docstring."""
        return self.nodes


def consolidate(points: Iterable[Sequence[float]]) -> ConsolidatedPoints:
    """
    Collapse a list of coordinates to the distinct 0.1 degree nodes they fall in.

    Order is preserved and the first coordinate seen for a node represents it, so
    a caller's list still reads back in the order they wrote it. Duplicates --
    exact, or merely in the same node -- collapse to one call.
    """
    normalised = [point_manifest.normalise(point) for point in points]
    if not normalised:
        msg = "at least one point is required"
        raise ValueError(msg)

    node_by_input: dict[Point, Point] = {}
    nodes: list[Point] = []
    for point in normalised:
        node = era5_bronze.node(point)
        node_by_input[point] = node
        if node not in nodes:
            nodes.append(node)

    grid = ConsolidatedPoints(nodes=tuple(nodes), node_by_input=node_by_input)
    logger.info(
        "consolidated %d requested point(s) -> %d grid node(s): %s",
        len(normalised),
        len(nodes),
        ", ".join(f"({lat:.1f}, {lon:.1f})" for lat, lon in nodes),
    )
    # The per-node detail is rendered by ``ClimateDatasets.describe_mapping``,
    # after the frames are read: only then is NSRDB's own cell coordinate known,
    # and a mapping that names one key but not the other is the kind of half
    # answer that sends a reader to a coordinate the frames are not keyed by.
    return grid


def group_by_node(grid: ConsolidatedPoints) -> dict[Point, list[Point]]:
    """The requested coordinates that landed on each node, node order preserved."""
    grouped: dict[Point, list[Point]] = {node: [] for node in grid.nodes}
    for point, node in grid.node_by_input.items():
        grouped[node].append(point)
    return {node: sorted(points) for node, points in grouped.items()}
