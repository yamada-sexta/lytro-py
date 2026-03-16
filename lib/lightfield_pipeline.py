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
    raw = RawImage.from_bytes(
        raw_bytes, info.width, info.height, info.raw.mosaic_tile, info.raw.mosaic_upper_left
    )
    return LightfieldImage.from_raw(raw, calibration)


def export_flat_png(
    raw_bytes: bytes,
    metadata_bytes: bytes,
    calibration: CalibrationData,
    output_path: str | Path,
) -> Path:
    meta = Metadata.from_bytes(metadata_bytes)
    info = meta.image_info()
    raw = RawImage.from_bytes(
        raw_bytes, info.width, info.height, info.raw.mosaic_tile, info.raw.mosaic_upper_left
    )
    lf = LightfieldImage.from_raw(raw, calibration)
    rgb_u16 = _apply_white_balance(lf.data, _extract_white_balance(meta))
    rgb_u16 = _tone_map_u16(rgb_u16)
    bgr = cv2.cvtColor(rgb_u16, cv2.COLOR_RGB2BGR)
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
    raw = RawImage.from_bytes(
        raw_bytes, info.width, info.height, info.raw.mosaic_tile, info.raw.mosaic_upper_left
    )
    rgb_u16 = _normalize_raw_rgb(raw.data, info, _extract_white_balance(meta))
    rgb_u16 = _tone_map_u16(rgb_u16)
    bgr = cv2.cvtColor(rgb_u16, cv2.COLOR_RGB2BGR)
    out_path = Path(output_path)
    if not cv2.imwrite(str(out_path), bgr):
        raise RuntimeError(f"Failed to write image to {out_path}")
    return out_path


def export_subaperture_tiled_png(
    raw_bytes: bytes,
    metadata_bytes: bytes,
    calibration: CalibrationData,
    output_path: str | Path,
    grid_size: int = 9,
) -> Path:
    if grid_size < 1 or grid_size % 2 == 0:
        raise ValueError("grid_size must be a positive odd integer")

    meta = Metadata.from_bytes(metadata_bytes)
    info = meta.image_info()
    raw = RawImage.from_bytes(
        raw_bytes, info.width, info.height, info.raw.mosaic_tile, info.raw.mosaic_upper_left
    )
    rgb_u16 = _normalize_raw_rgb(raw.data, info, _extract_white_balance(meta))

    tmp = _apply_calibration(rgb_u16, calibration)
    horizontal = calibration.array.grid.get_horizontal_lines()
    vertical = calibration.array.grid.get_vertical_lines()

    row_index, out_h = _build_subgrid_rows(horizontal)
    col_index, out_w = _build_subgrid_columns(vertical)
    if out_h == 0 or out_w == 0:
        raise RuntimeError("Calibration grid is empty; cannot export subaperture views.")

    pitch_y = _mean_subgrid_spacing(horizontal)
    pitch_x = _mean_subgrid_spacing(vertical)
    max_dx = 0.45 * pitch_x
    max_dy = 0.45 * pitch_y

    offsets_x = np.linspace(-max_dx, max_dx, grid_size)
    offsets_y = np.linspace(-max_dy, max_dy, grid_size)

    tile_h = out_h * grid_size
    tile_w = out_w * grid_size
    tiled = np.zeros((tile_h, tile_w, 3), dtype=np.uint16)

    for j, dy in enumerate(offsets_y):
        for i, dx in enumerate(offsets_x):
            view = _sample_subaperture(
                tmp,
                horizontal,
                vertical,
                row_index,
                col_index,
                dx,
                dy,
                out_h,
                out_w,
            )
            y0 = j * out_h
            x0 = i * out_w
            tiled[y0 : y0 + out_h, x0 : x0 + out_w] = view

    tiled = _tone_map_u16(tiled)
    bgr = cv2.cvtColor(tiled, cv2.COLOR_RGB2BGR)
    out_path = Path(output_path)
    if not cv2.imwrite(str(out_path), bgr):
        raise RuntimeError(f"Failed to write image to {out_path}")
    return out_path


def _normalize_raw_rgb(rgb: np.ndarray, info, wb_gains: np.ndarray | None) -> np.ndarray:
    rgb_u16 = rgb.astype(np.uint16)
    if info.raw.right_shift > 0:
        rgb_u16 = np.right_shift(rgb_u16, info.raw.right_shift).astype(np.uint16)

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

    rgb_f = rgb_u16.astype(np.float32)
    rgb_f = (rgb_f - offset) / scale
    rgb_f = np.clip(rgb_f, 0.0, 1.0)
    rgb_u16 = (rgb_f * 65535.0).astype(np.uint16)
    if wb_gains is not None:
        rgb_u16 = _apply_white_balance(rgb_u16, wb_gains)
    return rgb_u16


def _apply_calibration(rgb: np.ndarray, calibration: CalibrationData) -> np.ndarray:
    angle_deg = calibration.array.rotation * 180.0 / np.pi
    rotation = cv2.getRotationMatrix2D((0.0, 0.0), angle_deg, 1.0)
    translation = np.eye(3, dtype=np.float64)
    translation[0, 2] = calibration.array.translation[0]
    translation[1, 2] = calibration.array.translation[1]
    transform = rotation @ translation
    return cv2.warpAffine(rgb, transform, (rgb.shape[1], rgb.shape[0]))


def _mean_subgrid_spacing(lines) -> float:
    last = {}
    diffs = []
    for line in lines:
        subgrid = line.subgrid
        if subgrid in last:
            diffs.append(abs(line.position - last[subgrid]))
        last[subgrid] = line.position
    if not diffs:
        return 1.0
    return float(np.mean(diffs))


def _sample_subaperture(
    tmp: np.ndarray,
    horizontal,
    vertical,
    row_index: list[int],
    col_index: list[int],
    dx: float,
    dy: float,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    view = np.zeros((out_h, out_w, 3), dtype=np.uint16)
    for v_idx, vline in enumerate(vertical):
        for h_idx, hline in enumerate(horizontal):
            if hline.subgrid == vline.subgrid:
                src_x = int(round(vline.position + dx))
                src_y = int(round(hline.position + dy))
                out_x = row_index[h_idx]
                out_y = col_index[v_idx]
                if (
                    0 <= src_x < tmp.shape[1]
                    and 0 <= src_y < tmp.shape[0]
                    and 0 <= out_x < out_h
                    and 0 <= out_y < out_w
                ):
                    view[out_x, out_y] = tmp[src_y, src_x]
    return view


def _build_subgrid_rows(horizontal) -> tuple[list[int], int]:
    row_index: list[int] = []
    counts = {}
    for hline in horizontal:
        subgrid = hline.subgrid
        idx = counts.get(subgrid, 0)
        row_index.append(idx)
        counts[subgrid] = idx + 1
    out_h = max(counts.values(), default=0)
    return row_index, out_h


def _build_subgrid_columns(vertical) -> tuple[list[int], int]:
    col_index: list[int] = []
    counts = {}
    for vline in vertical:
        subgrid = vline.subgrid
        idx = counts.get(subgrid, 0)
        col_index.append(idx)
        counts[subgrid] = idx + 1
    out_w = max(counts.values(), default=0)
    return col_index, out_w


def _extract_white_balance(meta: Metadata) -> np.ndarray | None:
    try:
        color = meta.raw_json["master"]["picture"]["frameArray"][0]["frame"]["metadata"][
            "image"
        ]["color"]
        wb = color.get("whiteBalanceGain", {})
        r = float(wb.get("r", 1.0))
        gr = float(wb.get("gr", 1.0))
        gb = float(wb.get("gb", 1.0))
        b = float(wb.get("b", 1.0))
        g = 0.5 * (gr + gb)
        gains = np.array([r, g, b], dtype=np.float32)
        if np.allclose(gains, 1.0):
            return None
        return gains
    except Exception:
        return None


def _apply_white_balance(rgb: np.ndarray, gains: np.ndarray | None) -> np.ndarray:
    if gains is None:
        return rgb
    rgb_f = rgb.astype(np.float32)
    rgb_f *= gains.reshape((1, 1, 3))
    rgb_f = np.clip(rgb_f, 0.0, 65535.0)
    return rgb_f.astype(np.uint16)


def _tone_map_u16(rgb: np.ndarray) -> np.ndarray:
    if rgb.size == 0:
        return rgb
    # Use a luminance-based percentile stretch for a gentle normalization.
    rgb_f = rgb.astype(np.float32)
    lum = 0.2126 * rgb_f[..., 0] + 0.7152 * rgb_f[..., 1] + 0.0722 * rgb_f[..., 2]
    sample = lum[::4, ::4].reshape(-1)
    if sample.size == 0:
        sample = lum.reshape(-1)
    lo = float(np.percentile(sample, 1.0))
    hi = float(np.percentile(sample, 99.5))
    if hi <= lo + 1.0:
        return rgb
    scale = 1.0 / (hi - lo)
    rgb_f = (rgb_f - lo) * scale
    rgb_f = np.clip(rgb_f, 0.0, 1.0)
    return (rgb_f * 65535.0).astype(np.uint16)


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
