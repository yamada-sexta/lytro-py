from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .subgrid import SubGrid


@dataclass
class Line:
    subgrid: SubGrid = SubGrid.SUBGRID_A
    position: float = 0.0

    def to_json(self) -> dict:
        return {"subgrid": int(self.subgrid), "position": float(self.position)}

    @classmethod
    def from_json(cls, data: dict) -> "Line":
        return cls(SubGrid(int(data["subgrid"])), float(data["position"]))


@dataclass
class LineGrid:
    horizonal_lines: List[Line] = field(default_factory=list)
    vertical_lines: List[Line] = field(default_factory=list)

    def get_horizontal_lines(self) -> List[Line]:
        return self.horizonal_lines

    def get_vertical_lines(self) -> List[Line]:
        return self.vertical_lines

    def to_json(self) -> dict:
        return {
            "horizontal": [line.to_json() for line in self.horizonal_lines],
            "vertical": [line.to_json() for line in self.vertical_lines],
        }

    @classmethod
    def from_json(cls, data: dict) -> "LineGrid":
        horizontal = [Line.from_json(line) for line in data.get("horizontal", [])]
        vertical = [Line.from_json(line) for line in data.get("vertical", [])]
        return cls(horizontal, vertical)
