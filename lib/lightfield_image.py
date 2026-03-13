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

        horizontal = calibration.array.grid.horizontal
        vertical = calibration.array.grid.vertical
        out_h = len(horizontal)
        out_w = len(vertical) // 2
        out = np.zeros((out_h, out_w, 3), dtype=np.uint16)

        for y, vline in enumerate(vertical):
            for x, hline in enumerate(horizontal):
                if hline.subgrid == vline.subgrid:
                    src_x = int(round(vline.position))
                    src_y = int(round(hline.position))
                    if (
                        0 <= src_x < tmp.shape[1]
                        and 0 <= src_y < tmp.shape[0]
                        and 0 <= y // 2 < out_w
                    ):
                        out[x, y // 2] = tmp[src_y, src_x]
        return cls(out)
