from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

from .subgrid import SubGrid


@dataclass
class Point:
    position: Tuple[float, float]
    horizontal_line: int = 0
    vertical_line: int = 0


@dataclass
class Line:
    subgrid: SubGrid = SubGrid.SUBGRID_A
    line: List[Point] = field(default_factory=list)


class PointGrid:
    CONSTRUCT_LIM = 20
    MAX_DIFF = 3.0

    def __init__(self) -> None:
        self.accumulator: List[Point] = []
        self.storage: List[Point] = []
        self.lines_horizontal: List[Line] = []
        self.lines_vertical: List[Line] = []

    def add_point(self, point: Tuple[float, float]) -> None:
        stored = Point(point)
        self.storage.append(stored)
        self.accumulator.append(stored)

    def is_empty(self) -> bool:
        return not self.lines_horizontal and not self.lines_vertical

    def get_horizontal_lines(self) -> List[Line]:
        return self.lines_horizontal

    def get_vertical_lines(self) -> List[Line]:
        return self.lines_vertical

    def _lower_bound(self, keys: List[float], value: float) -> int:
        lo, hi = 0, len(keys)
        while lo < hi:
            mid = (lo + hi) // 2
            if keys[mid] < value:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _map_add_construct(self, keys: List[float], lines: List[Line], position: float, point: Point) -> None:
        if not keys:
            keys.append(position)
            lines.append(Line(line=[point]))
            return
        idx = self._lower_bound(keys, position)
        lb = idx - 1 if idx > 0 else None
        ub = idx if idx < len(keys) else None
        diff_lb = abs(keys[lb] - position) if lb is not None else float("inf")
        diff_ub = abs(keys[ub] - position) if ub is not None else float("inf")
        use_idx = lb if diff_lb < diff_ub else ub
        if use_idx is None or abs(keys[use_idx] - position) > self.MAX_DIFF:
            keys.insert(idx, position)
            lines.insert(idx, Line(line=[point]))
            return
        line = lines.pop(use_idx)
        keys.pop(use_idx)
        line.line.append(point)
        insert_idx = self._lower_bound(keys, position)
        keys.insert(insert_idx, position)
        lines.insert(insert_idx, line)

    def _map_add(self, keys: List[float], lines: List[Line], position: float, point: Point) -> None:
        if not keys:
            return
        idx = self._lower_bound(keys, position)
        lb = idx - 1 if idx > 0 else None
        ub = idx if idx < len(keys) else None
        diff_lb = abs(keys[lb] - position) if lb is not None else float("inf")
        diff_ub = abs(keys[ub] - position) if ub is not None else float("inf")
        use_idx = lb if diff_lb < diff_ub else ub
        if use_idx is None or abs(keys[use_idx] - position) >= self.MAX_DIFF:
            return
        line = lines.pop(use_idx)
        keys.pop(use_idx)
        line.line.append(point)
        insert_idx = self._lower_bound(keys, position)
        keys.insert(insert_idx, position)
        lines.insert(insert_idx, line)

    def _horizontal_line_inserter(self, start: int, end: int, inserter: Callable[[Point], None]) -> None:
        step = 1 if end > start else -1
        i = start
        while i != end:
            inserter(self.accumulator[i])
            i += step

    def _vertical_line_constructor(
        self,
        start: int,
        end: int,
        inserter_odd: Callable[[Point], None],
        inserter_even: Callable[[Point], None],
    ) -> None:
        step = 1 if end > start else -1
        i = start
        while i != end:
            tmp_line = self.lines_horizontal[i]
            if i & 1:
                for point in tmp_line.line:
                    inserter_even(point)
            else:
                for point in tmp_line.line:
                    inserter_odd(point)
            i += step

    def finalize(self) -> None:
        if not self.accumulator:
            return

        # Construct horizontal lines
        tmp_keys: List[float] = []
        tmp_lines: List[Line] = []

        construct_start = len(self.accumulator) // 3
        construct_start_pos = self.accumulator[construct_start].position[1]
        i = construct_start
        while i < len(self.accumulator) and self.accumulator[i].position[1] < construct_start_pos + self.CONSTRUCT_LIM:
            self._map_add_construct(tmp_keys, tmp_lines, self.accumulator[i].position[0], self.accumulator[i])
            i += 1
        while i < len(self.accumulator):
            self._map_add(tmp_keys, tmp_lines, self.accumulator[i].position[0], self.accumulator[i])
            i += 1

        # rekey by first point and add preceding points in reverse
        tmp_pairs = list(zip(tmp_keys, tmp_lines))
        tmp_keys = [line.line[0].position[0] for _, line in tmp_pairs]
        tmp_lines = [line for _, line in tmp_pairs]
        tmp_pairs = sorted(zip(tmp_keys, tmp_lines), key=lambda p: p[0])
        tmp_keys = [k for k, _ in tmp_pairs]
        tmp_lines = [l for _, l in tmp_pairs]

        i = construct_start
        while i >= 0:
            self._map_add(tmp_keys, tmp_lines, self.accumulator[i].position[0], self.accumulator[i])
            i -= 1

        for line in tmp_lines:
            line.line.sort(key=lambda p: p.position[1])

        self.lines_horizontal = []
        for idx, line in enumerate(tmp_lines):
            line.subgrid = SubGrid.SUBGRID_A if (idx & 1) == 0 else SubGrid.SUBGRID_B
            self.lines_horizontal.append(line)

        # Construct vertical lines
        tmp_keys_odd: List[float] = []
        tmp_lines_odd: List[Line] = []
        tmp_keys_even: List[float] = []
        tmp_lines_even: List[Line] = []

        construct_start = len(self.lines_horizontal) // 3
        self._vertical_line_constructor(
            construct_start,
            min(construct_start + 6, len(self.lines_horizontal)),
            lambda p: self._map_add_construct(tmp_keys_odd, tmp_lines_odd, p.position[1], p),
            lambda p: self._map_add_construct(tmp_keys_even, tmp_lines_even, p.position[1], p),
        )
        self._vertical_line_constructor(
            construct_start + 6,
            len(self.lines_horizontal),
            lambda p: self._map_add(tmp_keys_odd, tmp_lines_odd, p.position[1], p),
            lambda p: self._map_add(tmp_keys_even, tmp_lines_even, p.position[1], p),
        )

        tmp_pairs = list(zip(tmp_keys_odd, tmp_lines_odd))
        tmp_keys_odd = [line.line[0].position[1] for _, line in tmp_pairs]
        tmp_lines_odd = [line for _, line in tmp_pairs]
        tmp_pairs = sorted(zip(tmp_keys_odd, tmp_lines_odd), key=lambda p: p[0])
        tmp_keys_odd = [k for k, _ in tmp_pairs]
        tmp_lines_odd = [l for _, l in tmp_pairs]

        tmp_pairs = list(zip(tmp_keys_even, tmp_lines_even))
        tmp_keys_even = [line.line[0].position[1] for _, line in tmp_pairs]
        tmp_lines_even = [line for _, line in tmp_pairs]
        tmp_pairs = sorted(zip(tmp_keys_even, tmp_lines_even), key=lambda p: p[0])
        tmp_keys_even = [k for k, _ in tmp_pairs]
        tmp_lines_even = [l for _, l in tmp_pairs]

        self._vertical_line_constructor(
            construct_start,
            -1,
            lambda p: self._map_add(tmp_keys_odd, tmp_lines_odd, p.position[1], p),
            lambda p: self._map_add(tmp_keys_even, tmp_lines_even, p.position[1], p),
        )

        for line in tmp_lines_odd:
            line.line.sort(key=lambda p: p.position[0])
        for line in tmp_lines_even:
            line.line.sort(key=lambda p: p.position[0])

        self.lines_vertical = []
        it_odd = 0
        it_even = 0
        while it_odd < len(tmp_keys_odd) or it_even < len(tmp_keys_even):
            if it_odd == len(tmp_keys_odd):
                line = tmp_lines_even[it_even]
                line.subgrid = SubGrid.SUBGRID_B
                self.lines_vertical.append(line)
                it_even += 1
            elif it_even == len(tmp_keys_even):
                line = tmp_lines_odd[it_odd]
                line.subgrid = SubGrid.SUBGRID_A
                self.lines_vertical.append(line)
                it_odd += 1
            elif tmp_keys_odd[it_odd] < tmp_keys_even[it_even]:
                line = tmp_lines_odd[it_odd]
                line.subgrid = SubGrid.SUBGRID_A
                self.lines_vertical.append(line)
                it_odd += 1
            else:
                line = tmp_lines_even[it_even]
                line.subgrid = SubGrid.SUBGRID_B
                self.lines_vertical.append(line)
                it_even += 1

        # prune points not present in both line sets
        points_horizontal = {id(p) for line in self.lines_horizontal for p in line.line}
        self.storage = [p for p in self.storage if id(p) in points_horizontal]
        points_vertical = {id(p) for line in self.lines_vertical for p in line.line}
        for line in self.lines_horizontal:
            line.line[:] = [p for p in line.line if id(p) in points_vertical]
        self.storage = [p for p in self.storage if id(p) in points_vertical]

        # set line indices
        for i, line in enumerate(self.lines_vertical):
            for point in line.line:
                point.vertical_line = i
        for i, line in enumerate(self.lines_horizontal):
            for point in line.line:
                point.horizontal_line = i
