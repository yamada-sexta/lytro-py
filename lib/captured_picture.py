from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from lib.lightfield_pipeline import export_flat_png, load_calibration
from lib.lytro_device import LytroDevice, PictureEntry


@dataclass
class CapturedPicture:
    entry: PictureEntry
    metadata_bytes: bytes
    raw_bytes: bytes
    thumbnail_bytes: bytes
    _thumbnail_bgr: np.ndarray | None = None
    _thumbnail_gray8: np.ndarray | None = None

    THUMB_WIDTH = 128
    THUMB_HEIGHT = 128

    @classmethod
    async def create(
        cls, device: LytroDevice, entry: PictureEntry
    ) -> "CapturedPicture":
        metadata_bytes, raw_bytes, thumbnail_bytes = await cls._fetch_all(device, entry)
        return cls(
            entry=entry,
            metadata_bytes=metadata_bytes,
            raw_bytes=raw_bytes,
            thumbnail_bytes=thumbnail_bytes,
        )

    @staticmethod
    async def _fetch_all(
        device: LytroDevice, entry: PictureEntry
    ) -> tuple[bytes, bytes, bytes]:
        metadata = await device.get_file(entry.metadata_path)
        raw_data = await device.get_file(entry.raw_path)
        thumbnail = await device.get_file(entry.thumbnail_path)
        return metadata, raw_data, thumbnail

    def metadata_text(self) -> str:
        return self.metadata_bytes.decode("utf-8", errors="ignore")

    def thumbnail_bgr(self) -> np.ndarray | None:
        if self._thumbnail_bgr is None and self.thumbnail_bytes:
            gray = self.thumbnail_gray8()
            if gray is None:
                return None
            self._thumbnail_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return self._thumbnail_bgr

    def thumbnail_gray8(self) -> np.ndarray | None:
        if self._thumbnail_gray8 is not None:
            return self._thumbnail_gray8
        if not self.thumbnail_bytes:
            return None
        expected = self.THUMB_WIDTH * self.THUMB_HEIGHT * 2
        if len(self.thumbnail_bytes) < expected:
            raise RuntimeError(
                f"Thumbnail data too small: {len(self.thumbnail_bytes)} bytes"
            )
        raw = np.frombuffer(self.thumbnail_bytes[:expected], dtype="<u2")
        gray16 = raw.reshape((self.THUMB_HEIGHT, self.THUMB_WIDTH))
        # Match Lyli: qRgb(tmpVal, tmpVal, tmpVal) keeps only low 8 bits.
        self._thumbnail_gray8 = (gray16 & 0x00FF).astype(np.uint8)
        return self._thumbnail_gray8

    def save_metadata(self, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.write_bytes(self.metadata_bytes)
        return path

    def save_raw(self, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.write_bytes(self.raw_bytes)
        return path

    def save_thumbnail_bytes(self, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.write_bytes(self.thumbnail_bytes)
        return path

    def save_thumbnail_image(self, output_path: str | Path) -> Path:
        path = Path(output_path)
        image = self.thumbnail_bgr()
        if image is None:
            raise RuntimeError("Thumbnail data could not be decoded")
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Failed to write image to {path}")
        return path

    def export_all(self, output_dir: str | Path) -> dict[str, Path]:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        base = self.entry.basename
        outputs = {
            "metadata": self.save_metadata(directory / f"{base}.TXT"),
            "thumbnail": self.save_thumbnail_bytes(directory / f"{base}.128"),
            "raw": self.save_raw(directory / f"{base}.RAW"),
        }
        return outputs

    def export_flat(
        self, calibration_path: str | Path, output_path: str | Path
    ) -> Path:
        calibration = load_calibration(calibration_path)
        return export_flat_png(
            self.raw_bytes, self.metadata_bytes, calibration, output_path
        )
