import asyncio

from lib.lytro_device import LytroDevice


async def main() -> int:
    camera = LytroDevice.find()
    if camera is None:
        print("Lytro camera not found.")
        return 1

    pictures = []
    first_picture = None
    first_thumb = None
    try:
        await camera.wait_ready()
        info = await camera.get_camera_information()
        pictures = await camera.get_picture_list()
        first_picture = pictures[0] if pictures else None
        first_thumb = (
            await camera.get_file(first_picture.thumbnail_path)
            if first_picture
            else None
        )
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
    for pic in pictures:
        captured = pic.captured_at.isoformat() if pic.captured_at else "unknown"
        print(f"{pic.basename} (captured at {captured})")
        print(f"  SHA1: {pic.sha1_hex}")
        print(f"  Path: {pic.path}")
        print(f"  Thumbnail path: {pic.thumbnail_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
