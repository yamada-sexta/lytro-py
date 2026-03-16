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


def export_raw_png(
    raw_bytes: bytes,
    metadata_bytes: bytes,
    output_path: str | Path,
) -> Path:
    meta = Metadata.from_bytes(metadata_bytes)
    info = meta.image_info()
    raw = RawImage.from_bytes(raw_bytes, info.width, info.height)
    rgb = raw.data.astype(np.uint16)

    if info.raw.right_shift > 0:
        rgb = np.right_shift(rgb, info.raw.right_shift).astype(np.uint16)

    black_r = info.raw.black["r"]
    black_g = (info.raw.black["gr"] + info.raw.black["gb"]) / 2.0
    black_b = info.raw.black["b"]
    white_r = info.raw.white["r"]
    white_g = (info.raw.white["gr"] + info.raw.white["gb"]) / 2.0
    white_b = info.raw.white["b"]

    scale = np.array(
        [white_r - black_r, white_g - black_g, white_b - black_b],
        dtype=np.float32,
    )
    offset = np.array([black_r, black_g, black_b], dtype=np.float32)
    scale[scale == 0] = 1.0

    rgb_f = rgb.astype(np.float32)
    rgb_f = (rgb_f - offset) / scale
    rgb_f = np.clip(rgb_f, 0.0, 1.0)
    rgb_u16 = (rgb_f * 65535.0).astype(np.uint16)

    bgr = cv2.cvtColor(rgb_u16, cv2.COLOR_RGB2BGR)
    out_path = Path(output_path)
    if not cv2.imwrite(str(out_path), bgr):
        raise RuntimeError(f"Failed to write image to {out_path}")
    return out_path


def process_directory(
    input_dir: str | Path,
    calibration_path: str | Path,
    output_dir: str | Path | None = None,
    write_raw_png: bool = False,
) -> list[Path]:
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = input_dir
    output_dir = Path(output_dir)
    calibration = load_calibration(calibration_path)
    outputs: list[Path] = []
    for raw_path in sorted(input_dir.glob("*.RAW")):
        meta_path = raw_path.with_suffix(".TXT")
        if not meta_path.exists():
            continue
        base = raw_path.stem
        out_path = output_dir / f"{base}-flat.png"
        export_flat_png(
            raw_path.read_bytes(), meta_path.read_bytes(), calibration, out_path
        )
        outputs.append(out_path)
        if write_raw_png:
            raw_out_path = output_dir / f"{base}-raw.png"
            export_raw_png(
                raw_path.read_bytes(), meta_path.read_bytes(), raw_out_path
            )
            outputs.append(raw_out_path)
    return outputs
