import asyncio
from pathlib import Path

from lib.captured_picture import CapturedPicture
from lib.lytro_device import LytroDevice


async def main() -> int:
    camera = LytroDevice.find()
    if camera is None:
        print("Lytro camera not found.")
        return 1

    pictures = []
    raw_list = b""
    try:
        await camera.wait_ready()
        info = await camera.get_camera_information()
        raw_list = await camera.get_picture_list_raw()
        pictures = await camera.get_picture_list()

        print(f"Vendor: {info['vendor']}")
        print(f"Product: {info['product']}")
        print(f"Revision: {info['revision']}")
        print(f"Serial: {info['serial']}")
        print(f"Firmware: {info['firmware']}")
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

        return 0
    finally:
        camera.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
