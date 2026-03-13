from __future__ import annotations

import json
from pathlib import Path

from lib.lyli_metadata import Metadata
from lib.raw_image import RawImage
from lib.calibration_data import CalibrationData

from .calibrator import Calibrator
from .fftpreprocessor import FFTPreprocessor
from .lensdetector import LensDetector
from .preprocessor import Preprocessor


def calibrate_directory(
    input_dir: str | Path,
    output_path: str | Path,
    use_fft_preprocessor: bool = True,
) -> CalibrationData:
    directory = Path(input_dir)
    if not directory.exists():
        raise FileNotFoundError(directory)

    raw_files = sorted(p for p in directory.glob("*.RAW"))
    if not raw_files:
        raise RuntimeError(f"No .RAW files found in {directory}")

    preprocessor = FFTPreprocessor() if use_fft_preprocessor else Preprocessor()
    detector = LensDetector(preprocessor)
    calibrator = Calibrator()

    grids_added = 0
    for raw_path in raw_files:
        base = raw_path.with_suffix("")
        meta_path = base.with_suffix(".TXT")
        if not meta_path.exists():
            continue

        metadata = Metadata.from_bytes(meta_path.read_bytes())
        raw_bytes = raw_path.read_bytes()
        info = metadata.image_info()
        raw_img = RawImage.from_bytes(raw_bytes, info.width, info.height)
        point_grid = detector.detect(raw_img.data)
        if point_grid.is_empty():
            continue
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
