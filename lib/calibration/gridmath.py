from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from .gridmapper import GridMapper
from .linegrid import LineGrid, Line

LIMIT_HORIZONTAL = 17.0
LIMIT_VERTICAL = 8.0


@dataclass
class LineEntry:
    line: Line
    counter: int = 1
    referees: List[Tuple[int, int]] = field(default_factory=list)


def _insert_line(
    mapping: dict[float, LineEntry],
    line: Line,
    referee: Tuple[int, int],
    limit: float,
) -> None:
    if not mapping:
        mapping[line.position] = LineEntry(line=line, counter=1, referees=[referee])
        return
    keys = sorted(mapping.keys())
    pos = line.position
    # find lower_bound
    idx = 0
    while idx < len(keys) and keys[idx] < pos:
        idx += 1
    ub_key = keys[idx] if idx < len(keys) else None
    lb_key = keys[idx - 1] if idx > 0 else None
    diff_lb = abs(lb_key - pos) if lb_key is not None else float("inf")
    diff_ub = abs(ub_key - pos) if ub_key is not None else float("inf")
    key = lb_key if diff_lb < diff_ub else ub_key
    if key is None or abs(key - pos) >= limit:
        mapping[pos] = LineEntry(line=line, counter=1, referees=[referee])
        return
    entry = mapping.pop(key)
    entry.referees.append(referee)
    denom = 1.0 / (entry.counter + 1.0)
    entry.line.position = entry.counter * denom * entry.line.position + pos * denom
    entry.counter += 1
    mapping[entry.line.position] = entry


def average_grids(grids: List[LineGrid]) -> Tuple[LineGrid, List[GridMapper]]:
    if not grids:
        return LineGrid([], []), []

    horizontal: dict[float, LineEntry] = {}
    vertical: dict[float, LineEntry] = {}

    for idx, line in enumerate(grids[0].get_horizontal_lines()):
        horizontal[line.position] = LineEntry(line=line, counter=1, referees=[(0, idx)])
    for idx, line in enumerate(grids[0].get_vertical_lines()):
        vertical[line.position] = LineEntry(line=line, counter=1, referees=[(0, idx)])

    for grid_idx, grid in enumerate(grids[1:], start=1):
        for line_idx, line in enumerate(grid.get_horizontal_lines()):
            _insert_line(horizontal, line, (grid_idx, line_idx), LIMIT_HORIZONTAL)
        for line_idx, line in enumerate(grid.get_vertical_lines()):
            _insert_line(vertical, line, (grid_idx, line_idx), LIMIT_VERTICAL)

    mappers = [
        GridMapper.with_sizes(
            len(grid.get_horizontal_lines()), len(grid.get_vertical_lines())
        )
        for grid in grids
    ]

    linegrid = LineGrid([], [])
    line_index = 0
    for key in sorted(horizontal.keys()):
        entry = horizontal[key]
        if entry.counter > len(grids) / 2:
            linegrid.horizonal_lines.append(entry.line)
            for grid_idx, line_idx in entry.referees:
                mappers[grid_idx].set_horizontal(line_idx, line_index)
            line_index += 1

    line_index = 0
    for key in sorted(vertical.keys()):
        entry = vertical[key]
        if entry.counter > len(grids) / 2:
            linegrid.vertical_lines.append(entry.line)
            for grid_idx, line_idx in entry.referees:
                mappers[grid_idx].set_vertical(line_idx, line_index)
            line_index += 1

    return linegrid, mappers
