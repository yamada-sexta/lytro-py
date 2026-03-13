from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class SubGrid(IntEnum):
    SUBGRID_A = 0
    SUBGRID_B = 1


@dataclass(frozen=True)
class Line:
    subgrid: SubGrid
    position: float


@dataclass(frozen=True)
class LineGrid:
    horizontal: list[Line]
    vertical: list[Line]

    @classmethod
    def from_json(cls, data: dict) -> "LineGrid":
        horizontal = [
            Line(SubGrid(line["subgrid"]), float(line["position"]))
            for line in data.get("horizontal", [])
        ]
        vertical = [
            Line(SubGrid(line["subgrid"]), float(line["position"]))
            for line in data.get("vertical", [])
        ]
        return cls(horizontal=horizontal, vertical=vertical)


@dataclass(frozen=True)
class ArrayParameters:
    grid: LineGrid
    translation: np.ndarray  # shape (2,)
    rotation: float

    @classmethod
    def from_json(cls, data: dict) -> "ArrayParameters":
        grid = LineGrid.from_json(data["grid"])
        translation = np.array(data["translation"], dtype=np.float64)
        rotation = float(data["rotation"])
        return cls(grid=grid, translation=translation, rotation=rotation)


@dataclass(frozen=True)
class LensParameters:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray

    @classmethod
    def from_json(cls, data: dict) -> "LensParameters":
        camera_matrix = np.array(data["cameraMatrix"], dtype=np.float64)
        dist_coeffs = np.array(data["distCoeffs"], dtype=np.float64)
        return cls(camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)


@dataclass(frozen=True)
class LensConfiguration:
    zoom_step: int
    focus_step: int

    @classmethod
    def from_json(cls, data: dict) -> "LensConfiguration":
        return cls(zoom_step=int(data["zoomStep"]), focus_step=int(data["focusStep"]))


@dataclass(frozen=True)
class CalibrationData:
    serial: str
    array: ArrayParameters
    lens: list[tuple[LensConfiguration, LensParameters]]

    @classmethod
    def from_json(cls, data: dict) -> "CalibrationData":
        serial = str(data["serial"])
        array = ArrayParameters.from_json(data["array"])
        lens = []
        for entry in data.get("lens", []):
            config = LensConfiguration.from_json(entry["configuration"])
            params = LensParameters.from_json(entry["parameters"])
            lens.append((config, params))
        return cls(serial=serial, array=array, lens=lens)
