from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GridMapper:
    horizontal_mapping: list[int] = field(default_factory=list)
    vertical_mapping: list[int] = field(default_factory=list)

    @classmethod
    def with_sizes(cls, horizontal_size: int, vertical_size: int) -> "GridMapper":
        return cls([0] * horizontal_size, [0] * vertical_size)

    def map_horizontal(self, index: int) -> int:
        return self.horizontal_mapping[index]

    def map_vertical(self, index: int) -> int:
        return self.vertical_mapping[index]

    def set_horizontal(self, from_index: int, to_index: int) -> None:
        self.horizontal_mapping[from_index] = to_index

    def set_vertical(self, from_index: int, to_index: int) -> None:
        self.vertical_mapping[from_index] = to_index
