from __future__ import annotations

import numpy as np
import cv2

from .lensdetector import Mask, PreprocessorInterface


def _compute_edge_threshold(edges: np.ndarray) -> int:
    return int(np.mean(edges))


class Preprocessor(PreprocessorInterface):
    def preprocess(self, gray: np.ndarray) -> np.ndarray:
        out_mask = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
        threshold = _compute_edge_threshold(out_mask)
        _, tmp = cv2.threshold(out_mask, threshold, Mask.OBJECT, cv2.THRESH_BINARY)

        kernel = np.array(
            [[0.125, 0.125, 0.125], [0.125, 0, 0.125], [0.125, 0.125, 0.125]],
            dtype=np.float32,
        )
        out_mask = cv2.filter2D(tmp, cv2.CV_8U, kernel, anchor=(1, 1))
        _, out_mask = cv2.threshold(out_mask, 95, Mask.OBJECT, cv2.THRESH_BINARY)

        out_mask = cv2.multiply(out_mask, tmp)
        out_mask = Mask.OBJECT - out_mask

        element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3), anchor=(1, 1))
        tmp = cv2.erode(out_mask, element, anchor=(1, 1), iterations=2)
        out_mask = cv2.dilate(tmp, element, anchor=(1, 1), iterations=1)

        return out_mask
