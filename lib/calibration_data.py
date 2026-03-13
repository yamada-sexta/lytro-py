from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lib.calibration.linegrid import LineGrid


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

    def to_json(self) -> dict:
        return {
            "grid": self.grid.to_json(),
            "translation": self.translation.tolist(),
            "rotation": float(self.rotation),
        }


@dataclass(frozen=True)
class LensParameters:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray

    @classmethod
    def from_json(cls, data: dict) -> "LensParameters":
        camera_matrix = np.array(data["cameraMatrix"], dtype=np.float64)
        dist_coeffs = np.array(data["distCoeffs"], dtype=np.float64)
        return cls(camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

    def to_json(self) -> dict:
        return {
            "cameraMatrix": self.camera_matrix.tolist(),
            "distCoeffs": self.dist_coeffs.tolist(),
        }


@dataclass(frozen=True)
class LensConfiguration:
    zoom_step: int
    focus_step: int

    @classmethod
    def from_json(cls, data: dict) -> "LensConfiguration":
        return cls(zoom_step=int(data["zoomStep"]), focus_step=int(data["focusStep"]))

    def to_json(self) -> dict:
        return {"zoomStep": int(self.zoom_step), "focusStep": int(self.focus_step)}


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

    def to_json(self) -> dict:
        return {
            "serial": self.serial,
            "array": self.array.to_json(),
            "lens": [
                {
                    "configuration": config.to_json(),
                    "parameters": params.to_json(),
                }
                for config, params in self.lens
            ],
        }
