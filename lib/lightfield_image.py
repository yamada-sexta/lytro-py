from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from lib.calibration_data import CalibrationData
from lib.raw_image import RawImage


@dataclass
class LightfieldImage:
    data: np.ndarray  # uint16, shape (H, W, 3)

    @classmethod
    def from_raw(
        cls, raw: RawImage, calibration: CalibrationData
    ) -> "LightfieldImage":
        tmp = raw.data.copy()
        angle_deg = calibration.array.rotation * 180.0 / np.pi
        rotation = cv2.getRotationMatrix2D((0.0, 0.0), angle_deg, 1.0)
        translation = np.eye(3, dtype=np.float64)
        translation[0, 2] = calibration.array.translation[0]
        translation[1, 2] = calibration.array.translation[1]
        transform = rotation @ translation
        tmp = cv2.warpAffine(tmp, transform, (tmp.shape[1], tmp.shape[0]))

        horizontal = calibration.array.grid.get_horizontal_lines()
        vertical = calibration.array.grid.get_vertical_lines()
        row_index: list[int] = []
        row_counts = {}
        for hline in horizontal:
            subgrid = hline.subgrid
            idx = row_counts.get(subgrid, 0)
            row_index.append(idx)
            row_counts[subgrid] = idx + 1
        out_h = max(row_counts.values(), default=0)

        col_index: list[int] = []
        col_counts = {}
        for vline in vertical:
            subgrid = vline.subgrid
            idx = col_counts.get(subgrid, 0)
            col_index.append(idx)
            col_counts[subgrid] = idx + 1
        out_w = max(col_counts.values(), default=0)
        out = np.zeros((out_h, out_w, 3), dtype=np.uint16)

        for y, vline in enumerate(vertical):
            for x, hline in enumerate(horizontal):
                if hline.subgrid == vline.subgrid:
                    src_x = int(round(vline.position))
                    src_y = int(round(hline.position))
                    if (
                        0 <= src_x < tmp.shape[1]
                        and 0 <= src_y < tmp.shape[0]
                        and 0 <= row_index[x] < out_h
                        and 0 <= col_index[y] < out_w
                    ):
                        out[row_index[x], col_index[y]] = tmp[src_y, src_x]
        return cls(out)
