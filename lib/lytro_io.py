from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
from typing import Iterable

import imageio.v2 as iio
import numpy as np

from lib.lyli_metadata import Metadata
from lib.raw_image import RawImage


@dataclass(frozen=True)
class LytroLoadResult:
    rgb: np.ndarray  # uint16, shape (H, W, 3)
    metadata: Metadata


def load_lytro_rgb(
    raw_input: bytes | str | Path,
    metadata_input: bytes | str | Path | None = None,
) -> LytroLoadResult:
    raw_path = _as_path(raw_input)
    metadata_bytes = _read_optional_bytes(metadata_input)
    metadata = Metadata.from_bytes(metadata_bytes) if metadata_bytes is not None else None

    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    try:
        raw_path, metadata_path = _ensure_pair_on_disk(raw_input, metadata_input)
        image, meta = _read_with_imageio(raw_path, metadata)
        if metadata is None:
            metadata = _metadata_from_imageio(meta)
        if metadata is None:
            raise RuntimeError(
                "Could not resolve Lytro metadata. Provide a .TXT/.JSON metadata file."
            )
        rgb = _coerce_to_rgb(image, metadata)
        return LytroLoadResult(rgb=rgb, metadata=metadata)
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def _as_path(value: bytes | str | Path) -> Path | None:
    if isinstance(value, (str, Path)):
        return Path(value)
    return None


def _read_optional_bytes(
    value: bytes | str | Path | None,
) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value
    return Path(value).read_bytes()


def _ensure_pair_on_disk(
    raw_input: bytes | str | Path,
    metadata_input: bytes | str | Path | None,
) -> tuple[Path, Path | None]:
    raw_path = _as_path(raw_input)
    metadata_path = _as_path(metadata_input) if metadata_input is not None else None

    if raw_path is not None and metadata_path is None and not isinstance(metadata_input, bytes):
        return raw_path, None

    if raw_path is not None and metadata_path is not None:
        if raw_path.parent == metadata_path.parent and raw_path.stem == metadata_path.stem:
            return raw_path, metadata_path

    temp_dir = tempfile.TemporaryDirectory()
    tmp_root = Path(temp_dir.name)
    suffix = raw_path.suffix if raw_path is not None else ".RAW"
    tmp_raw = tmp_root / f"image{suffix}"
    if raw_path is not None:
        shutil.copy2(raw_path, tmp_raw)
    else:
        tmp_raw.write_bytes(raw_input)

    tmp_meta: Path | None = None
    if metadata_input is not None:
        tmp_meta = tmp_root / "image.TXT"
        if isinstance(metadata_input, bytes):
            tmp_meta.write_bytes(metadata_input)
        else:
            shutil.copy2(Path(metadata_input), tmp_meta)

    return tmp_raw, tmp_meta


def _select_plugin(raw_path: Path, metadata: Metadata | None) -> Iterable[str | None]:
    suffix = raw_path.suffix.lower()
    if suffix == ".lfp":
        return ("LYTRO-LFP", None)
    if suffix == ".lfr":
        return ("LYTRO-LFR", None)
    if suffix == ".raw":
        if metadata is not None:
            bits = metadata.image_info().raw.bits_per_pixel
            if bits <= 10:
                return ("LYTRO-ILLUM-RAW", "LYTRO-F01-RAW", None)
            return ("LYTRO-F01-RAW", "LYTRO-ILLUM-RAW", None)
        return ("LYTRO-F01-RAW", "LYTRO-ILLUM-RAW", None)
    return (None,)


def _read_with_imageio(
    raw_path: Path, metadata: Metadata | None
) -> tuple[np.ndarray, dict | None]:
    last_error: Exception | None = None
    for plugin in _select_plugin(raw_path, metadata):
        try:
            reader = (
                iio.get_reader(str(raw_path), format=plugin)
                if plugin
                else iio.get_reader(str(raw_path))
            )
            try:
                image = reader.get_data(0)
                meta = reader.get_meta_data()
            finally:
                reader.close()
            return np.asarray(image), meta
        except Exception as exc:  # pragma: no cover - best effort fallback
            last_error = exc
            continue
    if last_error is None:
        raise RuntimeError(f"Unable to read {raw_path} with imageio")
    raise last_error


def _metadata_from_imageio(meta: dict | None) -> Metadata | None:
    if not isinstance(meta, dict):
        return None
    if "master" in meta:
        return Metadata.from_dict(meta)
    for key in ("metadata", "meta", "json", "lytro", "lytro_metadata"):
        candidate = meta.get(key)
        if isinstance(candidate, dict) and "master" in candidate:
            return Metadata.from_dict(candidate)
    return None


def _coerce_to_rgb(image: np.ndarray, metadata: Metadata) -> np.ndarray:
    if image.ndim == 2:
        info = metadata.image_info()
        pattern = RawImage._bayer_pattern(info.raw.mosaic_tile, info.raw.mosaic_upper_left)
        bayer = image.astype(np.uint16, copy=False)
        rgb = RawImage._demosaic_cv(bayer, pattern)
        return rgb.astype(np.uint16, copy=False)
    if image.ndim == 3 and image.shape[2] >= 3:
        rgb = image[..., :3]
        if rgb.dtype != np.uint16:
            rgb = rgb.astype(np.uint16)
        return rgb
    raise ValueError(f"Unsupported image shape from imageio: {image.shape}")
