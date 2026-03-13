from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import cv2

from .lensdetector import Mask, PreprocessorInterface

HIGHPASS_CUTOFF = 10
HIGHPASS_CUTOFF_2 = HIGHPASS_CUTOFF * HIGHPASS_CUTOFF


class FFTPreprocessor(PreprocessorInterface):
    def preprocess(self, gray: NDArray[np.uint8]) -> NDArray[np.uint8]:
        plane0: NDArray[np.float32] = gray.astype(np.float32)
        plane1: NDArray[np.float32] = np.zeros_like(plane0)
        complex_i = cv2.merge([plane0, plane1])
        complex_i = cv2.dft(complex_i)

        planes = list(cv2.split(complex_i))
        for i in range(2):
            for y in range(HIGHPASS_CUTOFF, -HIGHPASS_CUTOFF - 1, -1):
                realy = cv2.borderInterpolate(y, planes[i].shape[0], cv2.BORDER_WRAP)
                x0 = int(round(np.sqrt(HIGHPASS_CUTOFF_2 - y * y)))
                for x in range(-x0, x0 + 1):
                    realx = cv2.borderInterpolate(
                        x, planes[i].shape[1], cv2.BORDER_WRAP
                    )
                    planes[i][realy, realx] = 0.0
        complex_i = cv2.merge(planes)

        inv_dft = cv2.idft(complex_i, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        inv_dft = np.asarray(inv_dft, dtype=np.float32)
        inv_dft_f32: NDArray[np.float32] = inv_dft
        norm: NDArray[np.float32] = np.empty_like(inv_dft, dtype=np.float32)
        cv2.normalize(inv_dft_f32, norm, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        out_mask: NDArray[np.uint8] = (norm * 255).astype(np.uint8)

        threshold = int(np.mean(out_mask)) + 20
        _, out_mask_mat = cv2.threshold(
            out_mask, threshold, Mask.OBJECT, cv2.THRESH_BINARY
        )
        out_mask = np.asarray(out_mask_mat, dtype=np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3), anchor=(1, 1))
        out_mask_mat = cv2.morphologyEx(
            out_mask, cv2.MORPH_OPEN, kernel, anchor=(1, 1), iterations=1
        )
        out_mask = np.asarray(out_mask_mat, dtype=np.uint8)

        return out_mask
