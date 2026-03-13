from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from lib.lytro_device import LytroDevice, PictureEntry


@dataclass
class CapturedPicture:
    entry: PictureEntry
    metadata_bytes: bytes
    raw_bytes: bytes
    thumbnail_bytes: bytes
    _thumbnail_bgr: np.ndarray | None = None

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
            arr = np.frombuffer(self.thumbnail_bytes, dtype=np.uint8)
            self._thumbnail_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return self._thumbnail_bgr

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
