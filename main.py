import asyncio

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
    finally:
        camera.close()

    print(f"Vendor: {info['vendor']}")
    print(f"Product: {info['product']}")
    print(f"Revision: {info['revision']}")
    print(f"Serial: {info['serial']}")
    print(f"Firmware: {info['firmware']}")
    print(f"Pictures: {len(pictures)}")
    # if first_picture:
    #     captured = (
    #         first_picture.captured_at.isoformat()
    #         if first_picture.captured_at
    #         else "unknown"
    #     )
    #     print(f"First image: {first_picture.basename} ({captured})")
    #     print(f"Thumbnail bytes: {len(first_thumb) if first_thumb else 0}")
    raw_time_map: dict[str, str] = {}
    if raw_list and len(raw_list) >= 12:
        line_len = int.from_bytes(raw_list[4:8], "little", signed=False)
        entry_offset = int.from_bytes(raw_list[8:12], "little", signed=False)
        pos = entry_offset * 8 + 12
        while pos + line_len <= len(raw_list):
            line = raw_list[pos : pos + line_len]
            file_base = line[8:16].split(b"\x00", 1)[0].decode(
                "ascii", errors="ignore"
            )
            file_id = int.from_bytes(line[20:24], "little", signed=False)
            basename = f"{file_base}{file_id:04d}"
            raw_time = line[96:120].split(b"\x00", 1)[0].decode(
                "ascii", errors="ignore"
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
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
