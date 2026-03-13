from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import cv2
import numpy as np

from .pointgrid import PointGrid


class Mask:
    EMPTY = 0
    PROCESSED = 128
    OBJECT = 255


class PreprocessorInterface:
    def preprocess(self, gray: np.ndarray) -> np.ndarray:
        raise NotImplementedError


MAX_LENS_SIZE = 15


def _find_centroid(image: np.ndarray, mask: np.ndarray, start: Tuple[int, int]) -> Tuple[float, float]:
    if start[0] == image.shape[1] - 1:
        return (0.0, 0.0)
    startx = start[0]
    endx = start[0] + 1
    y = start[1]
    m01 = 0.0
    m10 = 0.0
    total = 0.0

    maxy = min(y + MAX_LENS_SIZE, image.shape[0])
    maxx = min(startx + MAX_LENS_SIZE, image.shape[1])
    minx = max(startx - MAX_LENS_SIZE, 0)

    while y < maxy:
        pos = y * image.shape[1] + startx
        endpos = y * image.shape[1] + endx
        if mask.flat[pos] == Mask.OBJECT:
            oldstartx = startx
            tmppos = pos - 1
            x = startx - 1
            while x >= minx and mask.flat[tmppos] == Mask.OBJECT:
                m10 += y * image.flat[tmppos]
                m01 += x * image.flat[tmppos]
                total += image.flat[tmppos]
                mask.flat[tmppos] = Mask.PROCESSED
                tmppos -= 1
                startx -= 1
                x -= 1
            tmppos = pos
            x = oldstartx
            while x < maxx and mask.flat[tmppos] == Mask.OBJECT:
                m10 += y * image.flat[tmppos]
                m01 += x * image.flat[tmppos]
                total += image.flat[tmppos]
                mask.flat[tmppos] = Mask.PROCESSED
                tmppos += 1
                x += 1
            endx = x - 1
        else:
            tmppos = pos
            while mask.flat[tmppos] != Mask.OBJECT:
                if tmppos == endpos:
                    return (m01 / total, m10 / total) if total else (0.0, 0.0)
                tmppos += 1
                startx += 1
            x = startx
            while x < maxx and mask.flat[tmppos] == Mask.OBJECT:
                m10 += y * image.flat[tmppos]
                m01 += x * image.flat[tmppos]
                total += image.flat[tmppos]
                mask.flat[tmppos] = Mask.PROCESSED
                tmppos += 1
                x += 1
            endx = x - 1
        y += 1

    return (m01 / total, m10 / total) if total else (0.0, 0.0)


def _get_interpolated_color(image: np.ndarray, position: Tuple[float, float]) -> float:
    x, y = position
    if x < 0 or x > image.shape[1] - 1 or y < 0 or y > image.shape[0] - 1:
        return 0.0
    xx = int(np.floor(x))
    yy = int(np.floor(y))
    x0 = xx
    x1 = xx + 1
    y0 = yy
    y1 = yy + 1
    f00 = float(image[y0, x0])
    f01 = float(image[y0, x1])
    f10 = float(image[y1, x0])
    f11 = float(image[y1, x1])
    x0dif = x - x0
    x1dif = 1.0 - x0dif
    y0dif = y - y0
    y1dif = 1.0 - y0dif
    return f00 * y1dif * x1dif + f10 * x0dif * y1dif + f01 * x1dif * y0dif + f11 * x0dif * y0dif


def _compute_mask(radius: int) -> List[Tuple[float, float]]:
    mask = []
    r2 = radius * radius
    for y in range(radius, -radius - 1, -1):
        x0 = int(round(np.sqrt(r2 - y * y)))
        for x in range(-x0, x0 + 1):
            mask.append((float(x), float(y)))
    return mask


_OFFSET_LIST = [
    _compute_mask(3),
    _compute_mask(4),
    _compute_mask(5),
    _compute_mask(6),
]


def _refine_centroid(image: np.ndarray, start: Tuple[float, float]) -> Tuple[float, float]:
    estimate_x, estimate_y = start
    for mask in _OFFSET_LIST:
        m01 = 0.0
        m10 = 0.0
        total = 0.0
        for dx, dy in mask:
            pos = (estimate_x + dx, estimate_y + dy)
            pixel = _get_interpolated_color(image, pos)
            m10 += pos[1] * pixel
            m01 += pos[0] * pixel
            total += pixel
        if total != 0:
            estimate_x = m01 / total
            estimate_y = m10 / total
    return (estimate_x, estimate_y)


@dataclass
class LensDetector:
    preprocessor: PreprocessorInterface

    def detect(self, image: np.ndarray) -> PointGrid:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = np.clip(gray / 256.0, 0, 255).astype(np.uint8)

        mean = float(np.mean(gray[1620:1660, 1620:1660]))
        if mean < 16 or mean > 240:
            return PointGrid()

        mask = self.preprocessor.preprocess(gray)
        gray_t = gray.T.copy()
        mask_t = mask.T.copy()

        pointgrid = PointGrid()
        for row in range(mask_t.shape[0]):
            pixel = mask_t[row]
            for col in range(mask_t.shape[1]):
                if pixel[col] == Mask.OBJECT:
                    centroid = _find_centroid(gray_t, mask_t, (col, row))
                    centroid = _refine_centroid(gray_t, centroid)
                    pointgrid.add_point(centroid)
                    if (
                        0 <= int(round(centroid[1])) < mask_t.shape[0]
                        and 0 <= int(round(centroid[0])) < mask_t.shape[1]
                    ):
                        mask_t[int(round(centroid[1])), int(round(centroid[0]))] = 192

        pointgrid.finalize()
        return pointgrid
