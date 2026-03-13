from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from lib.calibration_data import CalibrationData
from lib.lightfield_image import LightfieldImage
from lib.lyli_metadata import Metadata
from lib.raw_image import RawImage


def load_calibration(path: str | Path) -> CalibrationData:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return CalibrationData.from_json(data)


def build_lightfield(
    raw_bytes: bytes, metadata_bytes: bytes, calibration: CalibrationData
) -> LightfieldImage:
    meta = Metadata.from_bytes(metadata_bytes)
    info = meta.image_info()
    raw = RawImage.from_bytes(raw_bytes, info.width, info.height)
    return LightfieldImage.from_raw(raw, calibration)


def export_flat_png(
    raw_bytes: bytes,
    metadata_bytes: bytes,
    calibration: CalibrationData,
    output_path: str | Path,
) -> Path:
    lf = build_lightfield(raw_bytes, metadata_bytes, calibration)
    bgr = cv2.cvtColor(lf.data, cv2.COLOR_RGB2BGR)
    out_path = Path(output_path)
    if not cv2.imwrite(str(out_path), bgr):
        raise RuntimeError(f"Failed to write image to {out_path}")
    return out_path
