from __future__ import annotations

import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from tqdm import tqdm

from lib.lyli_metadata import Metadata
from lib.raw_image import RawImage
from lib.calibration_data import CalibrationData

from .calibrator import Calibrator
from .fftpreprocessor import FFTPreprocessor
from .lensdetector import LensDetector
from .preprocessor import Preprocessor


def _process_file_for_calibration(
    raw_path_str: str, use_fft_preprocessor: bool
):
    raw_path = Path(raw_path_str)
    base = raw_path.with_suffix("")
    meta_path = base.with_suffix(".TXT")
    if not meta_path.exists():
        return None
    metadata = Metadata.from_bytes(meta_path.read_bytes())
    raw_bytes = raw_path.read_bytes()
    info = metadata.image_info()
    raw_img = RawImage.from_bytes(
        raw_bytes, info.width, info.height, info.raw.mosaic_tile, info.raw.mosaic_upper_left
    )
    preprocessor = FFTPreprocessor() if use_fft_preprocessor else Preprocessor()
    detector = LensDetector(preprocessor)
    point_grid = detector.detect(raw_img.data)
    if point_grid.is_empty():
        return None
    return point_grid, metadata


def calibrate_directory(
    input_dir: str | Path,
    output_path: str | Path,
    use_fft_preprocessor: bool = True,
    max_files: int | None = None,
    max_workers: int | None = None,
    use_processes: bool = True,
) -> CalibrationData:
    directory = Path(input_dir)
    if not directory.exists():
        raise FileNotFoundError(directory)

    raw_files = sorted(p for p in directory.glob("*.RAW"))
    if max_files is not None:
        raw_files = raw_files[:max_files]
    if not raw_files:
        raise RuntimeError(f"No .RAW files found in {directory}")

    calibrator = Calibrator()

    if max_workers is None:
        max_workers = os.cpu_count() or 1

    grids_added = 0
    Executor = ProcessPoolExecutor if use_processes and max_workers > 1 else ThreadPoolExecutor
    with Executor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_file_for_calibration, str(p), use_fft_preprocessor): p
            for p in raw_files
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Calibrating", unit="file"
        ):
            result = future.result()
            if result is None:
                continue
            point_grid, metadata = result
            calibrator.add_grid(point_grid, metadata)
            grids_added += 1

    if grids_added == 0:
        raise RuntimeError(
            f"No usable calibration grids detected in {directory}. "
            "Make sure calibration images are present."
        )
    calibration = calibrator.calibrate()
    output_path = Path(output_path)
    output_path.write_text(json.dumps(calibration.to_json(), indent=2), encoding="utf-8")
    return calibration
