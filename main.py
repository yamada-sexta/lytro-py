from lib.lytro_device import LytroDevice
import asyncio
import struct
import usb.core
import usb.util
from typing import TypedDict


async def main() -> int:
    camera = LytroDevice.find()
    if camera is None:
        print("Lytro camera not found.")
        return 1

    try:
        await camera.wait_ready()
        info = await camera.get_camera_information()
    finally:
        camera.close()

    print(f"Vendor: {info['vendor']}")
    print(f"Product: {info['product']}")
    print(f"Revision: {info['revision']}")
    print(f"Serial: {info['serial']}")
    print(f"Firmware: {info['firmware']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
