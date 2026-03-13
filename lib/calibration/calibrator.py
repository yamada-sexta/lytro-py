from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Tuple

import cv2
import numpy as np

from lib.calibration_data import CalibrationData, LensConfiguration, LensParameters, ArrayParameters
from lib.lyli_metadata import Metadata

from .exception import CameraDiffersException
from .gridmath import average_grids
from .mathutil import filtered_average, sgn
from .pointgrid import PointGrid
from .linegrid import LineGrid, Line

IMAGE_SIZE = 3280
SENSOR_SIZE = 0.0046


@dataclass(frozen=True)
class LensMeta:
    zoom_step: int
    focus_step: int
    focal_length: float


def _estimate_camera_matrix(lens: LensMeta) -> np.ndarray:
    focal_length_px = (lens.focal_length / SENSOR_SIZE) * IMAGE_SIZE
    center = IMAGE_SIZE / 2.0
    camera_matrix = np.zeros((3, 3), dtype=np.float64)
    camera_matrix[0, 0] = focal_length_px
    camera_matrix[1, 1] = focal_length_px
    camera_matrix[0, 2] = center
    camera_matrix[1, 2] = center
    camera_matrix[2, 2] = 1.0
    return camera_matrix


def _find_line_params(line: PointGrid.Line) -> np.ndarray:
    points = np.array([p.position for p in line.line], dtype=np.float32)
    line_params = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.001)
    return line_params.reshape(-1)


def _parametric_to_general(parametric: np.ndarray) -> np.ndarray:
    d = parametric[3] * parametric[0] - parametric[2] * parametric[1]
    return np.array([parametric[1], -parametric[0], d], dtype=np.float64)


def _rotate_general_line(line: np.ndarray, angle: float) -> np.ndarray:
    theta = math.atan2(line[1], line[0])
    p = -line[2] / math.sqrt(line[0] * line[0] + line[1] * line[1])
    return np.array([math.cos(theta + angle), math.sin(theta + angle), -p], dtype=np.float64)


def _calibrate_rotation(grid_list: List[PointGrid]) -> float:
    angles = []
    for grid in grid_list:
        local_angles = []
        for line in grid.get_vertical_lines():
            params = _find_line_params(line)
            optimal = np.array([1.0, 0.0])
            line_dir = np.array([params[0], params[1]])
            dot = float(np.dot(optimal, line_dir))
            dot = max(-1.0, min(1.0, dot))
            local_angles.append(math.acos(dot))
        angles.append(filtered_average(local_angles, 2.0))
    return filtered_average(angles, 2.0)


def _find_translation(
    lines: List[PointGrid.Line],
    direction: np.ndarray,
    angle: float,
    target: LineGrid,
    mapper,
) -> float:
    distances = []
    for line in lines:
        parametric = _find_line_params(line)
        general = _parametric_to_general(parametric)
        line_params = _rotate_general_line(general, angle)

        if direction[0] > 0.5:
            target_line = target.get_horizontal_lines()[
                mapper.map_horizontal(line.line[0].horizontal_line)
            ]
        else:
            target_line = target.get_vertical_lines()[
                mapper.map_vertical(line.line[0].vertical_line)
            ]

        target_params = np.array([direction[0], direction[1], -target_line.position], dtype=np.float64)
        intersect_params = np.array([direction[1], -direction[0], -IMAGE_SIZE / 2], dtype=np.float64)

        point1 = np.cross(line_params, intersect_params)
        point2 = np.cross(target_params, intersect_params)
        point1_e2 = np.array([point1[0] / point1[2], point1[1] / point1[2]])
        point2_e2 = np.array([point2[0] / point2[2], point2[1] / point2[2]])
        dif = point1_e2 - point2_e2
        sign = sgn(float(np.dot(dif, np.array([1.0, 1.0]))))
        distances.append(sign * float(np.linalg.norm(dif)))
    return filtered_average(distances, 2.0)


def _calibrate_translation(
    grid_list: List[PointGrid],
    angle: float,
    target: LineGrid,
    mappers,
) -> Tuple[float, float]:
    vertical_distances = []
    horizontal_distances = []
    for i, grid in enumerate(grid_list):
        vertical_distances.append(
            _find_translation(
                grid.get_horizontal_lines(), np.array([1.0, 0.0]), -angle, target, mappers[i]
            )
        )
        horizontal_distances.append(
            _find_translation(
                grid.get_vertical_lines(), np.array([0.0, 1.0]), -angle, target, mappers[i]
            )
        )
    vertical = filtered_average(vertical_distances, 2.0)
    horizontal = filtered_average(horizontal_distances, 2.0)
    return vertical, horizontal


def _linegrid_from_pointgrid(grid: PointGrid) -> LineGrid:
    horizontal = []
    for line in grid.get_horizontal_lines():
        mid_start = len(line.line) // 3
        mid_end = 2 * len(line.line) // 3
        xs = [line.line[i].position[0] for i in range(mid_start, mid_end)]
        position = sum(xs) / len(xs) if xs else 0.0
        horizontal.append(Line(subgrid=line.subgrid, position=position))

    vertical = []
    for line in grid.get_vertical_lines():
        mid_start = len(line.line) // 3
        mid_start = mid_start if (mid_start & 1) == 0 else mid_start - 1
        xs = [line.line[i].position[1] for i in range(mid_start, 2 * len(line.line) // 3, 2)]
        position = sum(xs) / len(xs) if xs else 0.0
        vertical.append(Line(subgrid=line.subgrid, position=position))

    return LineGrid(horizontal, vertical)


class Calibrator:
    def __init__(self) -> None:
        self._serial = ""
        self._pointgrids: List[PointGrid] = []
        self._cluster_map: Dict[LensMeta, List[int]] = {}

    def reset(self) -> None:
        self._serial = ""
        self._pointgrids = []
        self._cluster_map = {}

    def add_grid(self, pointgrid: PointGrid, metadata: Metadata) -> None:
        serial = metadata.private_serial()
        if self._serial and self._serial != serial:
            raise CameraDiffersException(
                f"Camera serial number differs, expected: {self._serial}, got: {serial}"
            )
        if not self._serial:
            self._serial = serial

        self._pointgrids.append(pointgrid)
        lens = metadata.lens_meta()
        key = LensMeta(
            zoom_step=lens["zoom_step"],
            focus_step=lens["focus_step"],
            focal_length=lens["focal_length"],
        )
        self._cluster_map.setdefault(key, []).append(len(self._pointgrids) - 1)

    def calibrate(self) -> CalibrationData:
        linegrids = [_linegrid_from_pointgrid(pg) for pg in self._pointgrids]
        target, mappers = average_grids(linegrids)

        rotation = _calibrate_rotation(self._pointgrids)
        translation = _calibrate_translation(self._pointgrids, -rotation, target, mappers)
        array_params = ArrayParameters(
            grid=target,
            translation=np.array([translation[0], translation[1]], dtype=np.float64),
            rotation=rotation,
        )

        lens_calib: List[tuple[LensConfiguration, LensParameters]] = []
        for lens_meta in self._cluster_map:
            camera_matrix = _estimate_camera_matrix(lens_meta)
            dist_coeffs = np.zeros((8, 1), dtype=np.float64)
            lens_params = LensParameters(camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
            lens_calib.append(
                (
                    LensConfiguration(zoom_step=lens_meta.zoom_step, focus_step=lens_meta.focus_step),
                    lens_params,
                )
            )

        return CalibrationData(serial=self._serial, array=array_params, lens=lens_calib)
