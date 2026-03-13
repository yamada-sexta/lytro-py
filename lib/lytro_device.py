import asyncio
import usb
from lib.usb_mass_storage import UsbMassStorage
from typing import TypedDict

LYTRO_VENDOR_ID = 0x24CF
LYTRO_PRODUCT_ID = 0x00A1


class CameraInfo(TypedDict):
    vendor: str
    product: str
    revision: str
    serial: str
    firmware: str


class LytroDevice(UsbMassStorage):
    """Lytro-specific camera implementation over USB Mass Storage."""

    @classmethod
    def find(cls) -> "LytroDevice | None":
        dev = usb.core.find(idVendor=LYTRO_VENDOR_ID, idProduct=LYTRO_PRODUCT_ID)
        if dev is None:
            return None
        if not isinstance(dev, usb.core.Device):
            raise RuntimeError("Multiple Lytro cameras found. This is not supported.")
        return cls(dev)

    async def wait_ready(self, retries: int = 100, delay_s: float = 0.1) -> None:
        cdb = bytes([0x00, 0x00, 0x00, 0x00, 0x00, 0x00])  # TEST UNIT READY
        for _ in range(retries):
            try:
                await self.command(cdb)
                return
            except Exception:
                await asyncio.sleep(delay_s)
        raise RuntimeError("Camera did not become ready in time")

    async def scsi_inquiry(self) -> tuple[str, str, str]:
        cdb = bytes([0x12, 0x00, 0x00, 0x00, 0x24, 0x00])  # INQUIRY, alloc 36
        data = await self.command(cdb, data_in_len=0x24)
        vendor = data[8:16].decode("ascii", errors="ignore").strip()
        product = data[16:32].decode("ascii", errors="ignore").strip()
        revision = data[32:36].decode("ascii", errors="ignore").strip()
        return vendor, product, revision

    async def download_data(self) -> bytes:
        await self.command(
            bytes(
                [0xC6, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
            ),
            data_in_len=65536,
        )
        result = bytearray()
        packet = 0
        while True:
            cdb = bytes(
                [
                    0xC4,
                    0x00,
                    0x01,
                    0x00,
                    0x00,
                    packet & 0xFF,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                ]
            )
            chunk = await self.command(cdb, data_in_len=65536)
            result.extend(chunk)
            if len(chunk) < 65536:
                break
            packet += 1
        return bytes(result)

    async def get_camera_information(self) -> CameraInfo:
        vendor, product, revision = await self.scsi_inquiry()
        await self.command(
            bytes(
                [0xC2, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
            )
        )
        info_response = await self.download_data()

        serial = (
            info_response[0x0100:]
            .split(b"\x00", 1)[0]
            .decode("ascii", errors="ignore")
            .strip()
        )
        firmware = (
            info_response[0x0200:]
            .split(b"\x00", 1)[0]
            .decode("ascii", errors="ignore")
            .strip()
        )

        return {
            "vendor": vendor,
            "product": product,
            "revision": revision,
            "serial": serial,
            "firmware": firmware,
        }
