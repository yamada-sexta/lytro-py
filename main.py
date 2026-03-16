from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from tap import Tap
from tqdm import tqdm

from lib.calibration.pipeline import calibrate_directory
from lib.captured_picture import CapturedPicture
from lib.lightfield_pipeline import (
    export_flat_png,
    export_raw_png,
    export_subaperture_tiled_png,
    load_calibration,
    process_directory,
)
from lib.lytro_device import LytroDevice, PictureEntry


class CalibrateArgs(Tap):
    input_dir: Path  # Directory with calibration .RAW/.TXT files
    output_path: Path = Path("calibration.json")  # Output calibration file

    def configure(self) -> None:
        self.add_argument("input_dir", type=Path)
        self.add_argument("output_path", nargs="?", default=Path("calibration.json"), type=Path)


class ProcessArgs(Tap):
    input_dir: Path  # Directory with .RAW/.TXT files
    calibration_path: Path = Path("calibration.json")  # Calibration file
    raw_png: bool = False  # Also write demosaiced raw PNGs

    def configure(self) -> None:
        self.add_argument("input_dir", type=Path)
        self.add_argument("calibration_path", nargs="?", default=Path("calibration.json"), type=Path)
        self.add_argument("--raw-png", action="store_true", dest="raw_png")


class ListRawArgs(Tap):
    input_dir: Path  # Directory to search for .RAW files

    def configure(self) -> None:
        self.add_argument("input_dir", type=Path)


class ListDeviceArgs(Tap):
    pass


class ExportRawsArgs(Tap):
    output_dir: Path  # Directory to write exported RAWs

    def configure(self) -> None:
        self.add_argument("output_dir", type=Path)


class ProcessDeviceArgs(Tap):
    output_dir: Path  # Directory to write processed PNGs
    calibration_path: Path = Path("calibration.json")  # Calibration file
    raw_png: bool = False  # Also write demosaiced raw PNGs

    def configure(self) -> None:
        self.add_argument("output_dir", type=Path)
        self.add_argument("calibration_path", nargs="?", default=Path("calibration.json"), type=Path)
        self.add_argument("--raw-png", action="store_true", dest="raw_png")


class ExportRawPngDeviceArgs(Tap):
    device_raw_path: str  # Device RAW path or basename (e.g. IMG_0003)
    output_path: Path  # Local PNG output path
    metadata_path: str | None = None  # Optional device TXT path

    def configure(self) -> None:
        self.add_argument("device_raw_path")
        self.add_argument("output_path", type=Path)
        self.add_argument("--metadata-path", dest="metadata_path", default=None)


class ExportSubapertureArgs(Tap):
    raw_path: Path  # Local RAW file path
    metadata_path: Path  # Local TXT metadata path
    output_path: Path  # Local PNG output path
    calibration_path: Path = Path("calibration.json")  # Calibration file
    grid: int = 9  # Odd grid size for subaperture views
    white_balance: bool = False  # Apply metadata white balance
    no_per_view_normalize: bool = False  # Disable per-view normalization
    no_aspect_correction: bool = False  # Disable aspect correction stretch

    def configure(self) -> None:
        self.add_argument("raw_path", type=Path)
        self.add_argument("metadata_path", type=Path)
        self.add_argument("output_path", type=Path)
        self.add_argument(
            "--calibration-path",
            dest="calibration_path",
            default=Path("calibration.json"),
            type=Path,
        )
        self.add_argument("--grid", type=int, default=9)
        self.add_argument(
            "--white-balance", action="store_true", dest="white_balance", default=False
        )
        self.add_argument(
            "--no-per-view-normalize",
            action="store_true",
            dest="no_per_view_normalize",
            default=False,
        )
        self.add_argument(
            "--no-color-correction",
            action="store_true",
            dest="no_color_correction",
            default=False,
        )
        self.add_argument(
            "--no-aspect-correction",
            action="store_true",
            dest="no_aspect_correction",
            default=False,
        )


class ExportSubapertureDeviceArgs(Tap):
    device_raw_path: str  # Device RAW path or basename (e.g. IMG_0003)
    output_path: Path  # Local PNG output path
    calibration_path: Path = Path("calibration.json")  # Calibration file
    metadata_path: str | None = None  # Optional device TXT path
    grid: int = 9  # Odd grid size for subaperture views
    white_balance: bool = False  # Apply metadata white balance
    no_per_view_normalize: bool = False  # Disable per-view normalization
    no_aspect_correction: bool = False  # Disable aspect correction stretch

    def configure(self) -> None:
        self.add_argument("device_raw_path")
        self.add_argument("output_path", type=Path)
        self.add_argument(
            "--calibration-path",
            dest="calibration_path",
            default=Path("calibration.json"),
            type=Path,
        )
        self.add_argument("--metadata-path", dest="metadata_path", default=None)
        self.add_argument("--grid", type=int, default=9)
        self.add_argument(
            "--white-balance", action="store_true", dest="white_balance", default=False
        )
        self.add_argument(
            "--no-per-view-normalize",
            action="store_true",
            dest="no_per_view_normalize",
            default=False,
        )
        self.add_argument(
            "--no-color-correction",
            action="store_true",
            dest="no_color_correction",
            default=False,
        )
        self.add_argument(
            "--no-aspect-correction",
            action="store_true",
            dest="no_aspect_correction",
            default=False,
        )


class Args(Tap):
    command: str | None = None

    def configure(self) -> None:
        self.add_subparsers(dest="command", help="sub-command help")
        self.add_subparser("calibrate", CalibrateArgs, help="generate calibration.json")
        self.add_subparser("process", ProcessArgs, help="process local RAWs")
        self.add_subparser("list-raw", ListRawArgs, help="list local .RAW files")
        self.add_subparser("list-device", ListDeviceArgs, help="list RAWs on device")
        self.add_subparser("export-raws", ExportRawsArgs, help="export RAWs from device")
        self.add_subparser("process-device", ProcessDeviceArgs, help="process RAWs from device")
        self.add_subparser(
            "export-raw-png-device",
            ExportRawPngDeviceArgs,
            help="export a single RAW from device to PNG",
        )
        self.add_subparser(
            "export-subaperture",
            ExportSubapertureArgs,
            help="export tiled subaperture PNG from local RAW/TXT",
        )
        self.add_subparser(
            "export-subaperture-device",
            ExportSubapertureDeviceArgs,
            help="export tiled subaperture PNG from device RAW",
        )


async def _list_device(camera: LytroDevice) -> list[PictureEntry]:
    await camera.wait_ready()
    return await camera.get_picture_list()


async def _resolve_device_paths(
    camera: LytroDevice,
    device_raw_path: str,
    metadata_path: str | None,
) -> tuple[str, str]:
    if metadata_path is not None:
        return device_raw_path, metadata_path
    if device_raw_path.upper().endswith(".RAW"):
        return device_raw_path, device_raw_path[:-4] + ".TXT"
    candidates = await _list_device(camera)
    for pic in candidates:
        if pic.basename == device_raw_path or pic.basename == device_raw_path.upper():
            return pic.raw_path, pic.metadata_path
    raise ValueError(
        f"Could not find RAW on device for basename: {device_raw_path}. "
        "Use a full device path or provide --metadata-path."
    )


def _get_command_name(args: Any) -> str | None:
    command = getattr(args, "command", None)
    if command:
        return command
    if isinstance(args, CalibrateArgs):
        return "calibrate"
    if isinstance(args, ProcessArgs):
        return "process"
    if isinstance(args, ListRawArgs):
        return "list-raw"
    if isinstance(args, ListDeviceArgs):
        return "list-device"
    if isinstance(args, ExportRawsArgs):
        return "export-raws"
    if isinstance(args, ProcessDeviceArgs):
        return "process-device"
    if isinstance(args, ExportRawPngDeviceArgs):
        return "export-raw-png-device"
    if isinstance(args, ExportSubapertureArgs):
        return "export-subaperture"
    if isinstance(args, ExportSubapertureDeviceArgs):
        return "export-subaperture-device"
    return None


async def _run_default() -> int:
    print("No command specified...")
    camera = LytroDevice.find()
    if camera is None:
        print("Lytro camera not found.")
        return 1

    pictures: list[PictureEntry] = []
    raw_list = b""
    try:
        print("Camera found. Gathering information...")
        await camera.wait_ready()
        print("Camera is ready. Fetching data...")
        info = await camera.get_camera_information()
        raw_list = await camera.get_picture_list_raw()
        pictures = await camera.get_picture_list()

        print(f"Vendor: {info.vendor}")
        print(f"Product: {info.product}")
        print(f"Revision: {info.revision}")
        print(f"Serial: {info.serial}")
        print(f"Firmware: {info.firmware}")
        print(f"Pictures: {len(pictures)}")

        raw_time_map: dict[str, str] = {}
        if raw_list and len(raw_list) >= 12:
            line_len = int.from_bytes(raw_list[4:8], "little", signed=False)
            entry_offset = int.from_bytes(raw_list[8:12], "little", signed=False)
            pos = entry_offset * 8 + 12
            while pos + line_len <= len(raw_list):
                line = raw_list[pos : pos + line_len]
                file_base = (
                    line[8:16].split(b"\x00", 1)[0].decode("ascii", errors="ignore")
                )
                file_id = int.from_bytes(line[20:24], "little", signed=False)
                basename = f"{file_base}{file_id:04d}"
                raw_time = (
                    line[96:120].split(b"\x00", 1)[0].decode("ascii", errors="ignore")
                )
                raw_time_map[basename] = raw_time
                pos += line_len

        for pic in pictures:
            captured = pic.captured_at.isoformat() if pic.captured_at else "unknown"
            print(f"{pic.basename} (captured at {captured})")
            print(f"  Raw time: {raw_time_map.get(pic.basename, 'missing')}")
            print(f"  SHA1: {pic.sha1_hex}")
            print(f"  Path: {pic.path}")
            print(f"  Thumbnail path: {pic.thumbnail_path}")

        if pictures:
            sample = await CapturedPicture.create(camera, pictures[0])
            output_dir = Path("output")
            outputs = sample.export_all(output_dir)
            print(f"Exported metadata: {outputs['metadata']}")
            print(f"Exported thumbnail: {outputs['thumbnail']}")
            print(f"Exported raw: {outputs['raw']}")
            try:
                thumb_path = output_dir / f"{sample.entry.basename}-thumb.png"
                sample.save_thumbnail_image(thumb_path)
                print(f"Exported thumbnail image: {thumb_path}")
            except RuntimeError as exc:
                print(f"Thumbnail decode failed: {exc}")
            calibration_path = Path("calibration.json")
            if not calibration_path.exists():
                print("Calibration file missing: calibration.json. Generating it now...")
                calib_dir = output_dir / "calibration"
                calib_dir.mkdir(parents=True, exist_ok=True)
                # Download calibration images from camera (C:\\T1CALIB\\MOD_0000..0061)
                downloaded = 0
                skipped = 0
                for i in tqdm(range(62), desc="Downloading calibration images", unit="image"):
                    name = f"MOD_{i:04d}"
                    raw_path = f"C:\\T1CALIB\\{name}.RAW"
                    txt_path = f"C:\\T1CALIB\\{name}.TXT"
                    try:
                        if (calib_dir / f"{name}.RAW").exists() and (
                            calib_dir / f"{name}.TXT"
                        ).exists():
                            print(
                                f"Calibration image {name} already exists. Skipping download."
                            )
                            downloaded += 1
                            continue
                        raw_bytes = await camera.get_file(raw_path)
                        txt_bytes = await camera.get_file(txt_path)
                        (calib_dir / f"{name}.RAW").write_bytes(raw_bytes)
                        (calib_dir / f"{name}.TXT").write_bytes(txt_bytes)
                        downloaded += 1
                    except Exception as exc:
                        print(f"Skipping calibration image {name}: {exc}")
                        skipped += 1
                print(f"Downloaded calibration images: {downloaded}, skipped: {skipped}")
                calibrate_directory(calib_dir, calibration_path)
                print(f"Wrote calibration file: {calibration_path}")

            if calibration_path.exists():
                flat_path = output_dir / f"{sample.entry.basename}-flat.png"
                sample.export_flat(calibration_path, flat_path)
                print(f"Exported flat image: {flat_path}")
                color_thumb_path = output_dir / f"{sample.entry.basename}-thumb.png"
                sample.save_color_thumbnail(calibration_path, color_thumb_path)
                print(f"Exported color thumbnail image: {color_thumb_path}")
            else:
                print("Calibration still missing after generation. Skipping flat export.")

        return 0
    finally:
        camera.close()


async def main() -> int:
    args = Args().parse_args()
    command = _get_command_name(args)

    if command == "calibrate":
        input_dir = Path(getattr(args, "input_dir"))
        output_path = Path(getattr(args, "output_path"))
        calibrate_directory(input_dir, output_path)
        print(f"Wrote calibration file: {output_path}")
        return 0

    if command == "process":
        input_dir = Path(getattr(args, "input_dir"))
        calibration_path = Path(getattr(args, "calibration_path"))
        write_raw_png = bool(getattr(args, "raw_png"))
        outputs = process_directory(
            input_dir, calibration_path, write_raw_png=write_raw_png
        )
        print(f"Generated {len(outputs)} images")
        return 0

    if command == "list-raw":
        input_dir = Path(getattr(args, "input_dir"))
        raw_files = sorted(input_dir.glob("*.RAW"))
        if not raw_files:
            print(f"No .RAW files found in {input_dir}")
            return 0
        for raw_path in raw_files:
            print(raw_path)
        return 0

    if command == "list-device":
        camera = LytroDevice.find()
        if camera is None:
            print("Lytro camera not found.")
            return 1
        try:
            print("Camera found. Fetching picture list...")
            pictures = await _list_device(camera)
            for pic in pictures:
                print(f"{pic.basename} -> {pic.raw_path}")
            print(f"Total: {len(pictures)}")
            return 0
        finally:
            camera.close()

    if command == "export-raws":
        output_dir = Path(getattr(args, "output_dir"))
        camera = LytroDevice.find()
        if camera is None:
            print("Lytro camera not found.")
            return 1
        try:
            print("Camera found. Downloading RAWs...")
            pictures = await _list_device(camera)
            output_dir.mkdir(parents=True, exist_ok=True)
            for pic in tqdm(pictures, desc="Exporting RAWs", unit="image"):
                captured = await CapturedPicture.create(camera, pic)
                captured.export_all(output_dir)
            print(f"Exported {len(pictures)} RAWs to {output_dir}")
            return 0
        finally:
            camera.close()

    if command == "process-device":
        output_dir = Path(getattr(args, "output_dir"))
        calibration_path = Path(getattr(args, "calibration_path"))
        write_raw_png = bool(getattr(args, "raw_png"))
        camera = LytroDevice.find()
        if camera is None:
            print("Lytro camera not found.")
            return 1
        try:
            print("Camera found. Downloading and processing images...")
            pictures = await _list_device(camera)
            output_dir.mkdir(parents=True, exist_ok=True)
            calibration = load_calibration(calibration_path)
            for pic in tqdm(pictures, desc="Processing images", unit="image"):
                metadata_bytes = await camera.get_file(pic.metadata_path)
                raw_bytes = await camera.get_file(pic.raw_path)
                base = pic.basename
                flat_path = output_dir / f"{base}-flat.png"
                export_flat_png(raw_bytes, metadata_bytes, calibration, flat_path)
                if write_raw_png:
                    raw_path = output_dir / f"{base}-raw.png"
                    export_raw_png(raw_bytes, metadata_bytes, raw_path)
            print(f"Generated {len(pictures)} images in {output_dir}")
            return 0
        finally:
            camera.close()

    if command == "export-raw-png-device":
        device_raw_path = str(getattr(args, "device_raw_path"))
        output_path = Path(getattr(args, "output_path"))
        metadata_path = getattr(args, "metadata_path")
        camera = LytroDevice.find()
        if camera is None:
            print("Lytro camera not found.")
            return 1
        try:
            print("Camera found. Downloading RAW...")
            await camera.wait_ready()
            raw_path, meta_path = await _resolve_device_paths(
                camera, device_raw_path, metadata_path
            )
            raw_bytes = await camera.get_file(raw_path)
            metadata_bytes = await camera.get_file(meta_path)
            export_raw_png(raw_bytes, metadata_bytes, output_path)
            print(f"Wrote PNG: {output_path}")
            return 0
        finally:
            camera.close()

    if command == "export-subaperture":
        raw_path = Path(getattr(args, "raw_path"))
        metadata_path = Path(getattr(args, "metadata_path"))
        output_path = Path(getattr(args, "output_path"))
        calibration_path = Path(getattr(args, "calibration_path"))
        grid = int(getattr(args, "grid"))
        apply_wb = bool(getattr(args, "white_balance"))
        per_view_normalize = not bool(getattr(args, "no_per_view_normalize"))
        apply_ccm = not bool(getattr(args, "no_color_correction"))
        apply_aspect = not bool(getattr(args, "no_aspect_correction"))
        calibration = load_calibration(calibration_path)
        export_subaperture_tiled_png(
            raw_path.read_bytes(),
            metadata_path.read_bytes(),
            calibration,
            output_path,
            grid_size=grid,
            apply_white_balance=apply_wb,
            apply_color_correction=apply_ccm,
            per_view_normalize=per_view_normalize,
            apply_aspect_correction=apply_aspect,
        )
        print(f"Wrote PNG: {output_path}")
        return 0

    if command == "export-subaperture-device":
        device_raw_path = str(getattr(args, "device_raw_path"))
        output_path = Path(getattr(args, "output_path"))
        calibration_path = Path(getattr(args, "calibration_path"))
        metadata_path = getattr(args, "metadata_path")
        grid = int(getattr(args, "grid"))
        apply_wb = bool(getattr(args, "white_balance"))
        per_view_normalize = not bool(getattr(args, "no_per_view_normalize"))
        apply_ccm = not bool(getattr(args, "no_color_correction"))
        apply_aspect = not bool(getattr(args, "no_aspect_correction"))
        calibration = load_calibration(calibration_path)
        camera = LytroDevice.find()
        if camera is None:
            print("Lytro camera not found.")
            return 1
        try:
            print("Camera found. Downloading RAW...")
            await camera.wait_ready()
            raw_path, meta_path = await _resolve_device_paths(
                camera, device_raw_path, metadata_path
            )
            raw_bytes = await camera.get_file(raw_path)
            metadata_bytes = await camera.get_file(meta_path)
            export_subaperture_tiled_png(
                raw_bytes,
                metadata_bytes,
                calibration,
                output_path,
                grid_size=grid,
                apply_white_balance=apply_wb,
                apply_color_correction=apply_ccm,
                per_view_normalize=per_view_normalize,
                apply_aspect_correction=apply_aspect,
            )
            print(f"Wrote PNG: {output_path}")
            return 0
        finally:
            camera.close()

    return await _run_default()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
