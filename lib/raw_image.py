from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class RawImage:
    data: np.ndarray  # uint16, shape (H, W, 3)

    @classmethod
    def from_bytes(cls, raw_bytes: bytes, width: int, height: int) -> "RawImage":
        rgb = np.zeros((height, width, 3), dtype=np.uint16)
        total_pixels = width * height
        pairs = total_pixels // 2
        expected_bytes = pairs * 3
        if len(raw_bytes) < expected_bytes:
            raise ValueError(
                f"RAW data too short: {len(raw_bytes)} bytes (expected {expected_bytes})"
            )

        buf = np.frombuffer(raw_bytes[:expected_bytes], dtype=np.uint8).reshape(
            (-1, 3)
        )
        # 12-bit packed: [b0 b1 b2] -> p0 = b0<<4 | (b1>>4), p1 = (b1&0x0F)<<8 | b2
        p0 = (buf[:, 0].astype(np.uint16) << 4) | (buf[:, 1] >> 4).astype(
            np.uint16
        )
        p1 = ((buf[:, 1] & 0x0F).astype(np.uint16) << 8) | buf[:, 2].astype(
            np.uint16
        )
        pixels = np.empty(total_pixels, dtype=np.uint16)
        pixels[0::2] = p0
        pixels[1::2] = p1

        bayer = pixels.reshape((height, width))

        # BGGR pattern:
        # B G B G
        # G R G R
        rgb[0::2, 0::2, 2] = bayer[0::2, 0::2]  # blue
        rgb[0::2, 1::2, 1] = bayer[0::2, 1::2]  # green
        rgb[1::2, 0::2, 1] = bayer[1::2, 0::2]  # green
        rgb[1::2, 1::2, 0] = bayer[1::2, 1::2]  # red

        cls._demosaic_bilinear(rgb)
        return cls(rgb)

    @staticmethod
    def _avg2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return ((a.astype(np.uint32) + b.astype(np.uint32)) // 2).astype(np.uint16)

    @staticmethod
    def _avg4(
        a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
    ) -> np.ndarray:
        return (
            (
                a.astype(np.uint32)
                + b.astype(np.uint32)
                + c.astype(np.uint32)
                + d.astype(np.uint32)
            )
            // 4
        ).astype(np.uint16)

    @classmethod
    def _demosaic_bilinear(cls, rgb: np.ndarray) -> None:
        rows, cols, _ = rgb.shape
        row_max = rows - 1
        col_max = cols - 1

        # red lines (y odd), green positions (x even)
        y = slice(1, row_max, 2)
        x = slice(2, col_max, 2)
        x_left = slice(1, col_max - 1, 2)
        x_right = slice(3, col_max + 1, 2)
        y_up = slice(0, row_max - 1, 2)
        y_down = slice(2, row_max + 1, 2)
        rgb[y, x, 0] = cls._avg2(rgb[y, x_left, 0], rgb[y, x_right, 0])
        rgb[y, x, 2] = cls._avg2(rgb[y_up, x, 2], rgb[y_down, x, 2])

        # blue lines (y even), green positions (x odd)
        y = slice(2, row_max, 2)
        x = slice(1, col_max, 2)
        y_up = slice(1, row_max - 1, 2)
        y_down = slice(3, row_max + 1, 2)
        x_left = slice(0, col_max - 1, 2)
        x_right = slice(2, col_max + 1, 2)
        rgb[y, x, 0] = cls._avg2(rgb[y_up, x, 0], rgb[y_down, x, 0])
        rgb[y, x, 2] = cls._avg2(rgb[y, x_left, 2], rgb[y, x_right, 2])

        # blue positions -> fill red + green
        y = slice(2, row_max, 2)
        x = slice(2, col_max, 2)
        y_up = slice(1, row_max - 1, 2)
        y_down = slice(3, row_max + 1, 2)
        x_left = slice(1, col_max - 1, 2)
        x_right = slice(3, col_max + 1, 2)
        rgb[y, x, 0] = cls._avg4(
            rgb[y_up, x_left, 0],
            rgb[y_up, x_right, 0],
            rgb[y_down, x_left, 0],
            rgb[y_down, x_right, 0],
        )
        rgb[y, x, 1] = cls._avg4(
            rgb[y, x_left, 1],
            rgb[y, x_right, 1],
            rgb[y_up, x, 1],
            rgb[y_down, x, 1],
        )

        # red positions -> fill blue + green
        y = slice(1, row_max, 2)
        x = slice(1, col_max, 2)
        y_up = slice(0, row_max - 1, 2)
        y_down = slice(2, row_max + 1, 2)
        x_left = slice(0, col_max - 1, 2)
        x_right = slice(2, col_max + 1, 2)
        rgb[y, x, 2] = cls._avg4(
            rgb[y_up, x_left, 2],
            rgb[y_up, x_right, 2],
            rgb[y_down, x_left, 2],
            rgb[y_down, x_right, 2],
        )
        rgb[y, x, 1] = cls._avg4(
            rgb[y, x_left, 1],
            rgb[y, x_right, 1],
            rgb[y_up, x, 1],
            rgb[y_down, x, 1],
        )

        # borders: copy nearest valid row/col
        rgb[0, :, :] = rgb[1, :, :]
        rgb[row_max, :, :] = rgb[row_max - 1, :, :]
        rgb[:, 0, :] = rgb[:, 1, :]
        rgb[:, col_max, :] = rgb[:, col_max - 1, :]
