import struct
import sys
import time

import usb.core
import usb.util


LYTRO_VENDOR_ID = 0x24CF
LYTRO_PRODUCT_ID = 0x00A1

DEFAULT_EP_OUT = 0x02
DEFAULT_EP_IN = 0x82
DEFAULT_INTERFACE = 0

CBW_SIGNATURE = 0x43425355  # "USBC"
CSW_SIGNATURE = 0x53425355  # "USBS"


class UsbMassStorage:
    def __init__(self, dev: usb.core.Device) -> None:
        self.dev = dev
        self._detach_kernel_driver()
        try:
            self.dev.set_configuration()
        except usb.core.USBError as exc:
            if exc.errno == 16:
                raise RuntimeError(
                    "USB resource busy. Another driver/app is using the camera. "
                    "Try unmounting the Lytro volume or detaching the kernel "
                    "driver, then re-run."
                ) from exc
            raise
        usb.util.claim_interface(self.dev, DEFAULT_INTERFACE)
        self.ep_out, self.ep_in = self._find_endpoints()
        self._tag = 1

    def _detach_kernel_driver(self) -> None:
        try:
            if self.dev.is_kernel_driver_active(DEFAULT_INTERFACE):
                self.dev.detach_kernel_driver(DEFAULT_INTERFACE)
        except NotImplementedError:
            return

    def _find_endpoints(self):
        cfg = self.dev.get_active_configuration()
        intf = cfg[(0, 0)]
        ep_out = usb.util.find_descriptor(
            intf,
            custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress)
            == usb.util.ENDPOINT_OUT,
        )
        ep_in = usb.util.find_descriptor(
            intf,
            custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress)
            == usb.util.ENDPOINT_IN,
        )
        if ep_out is None or ep_in is None:
            return DEFAULT_EP_OUT, DEFAULT_EP_IN
        return ep_out.bEndpointAddress, ep_in.bEndpointAddress

    def _next_tag(self) -> int:
        tag = self._tag
        self._tag = (self._tag + 1) & 0xFFFFFFFF
        if self._tag == 0:
            self._tag = 1
        return tag

    def _send_cbw(self, cdb: bytes, data_len: int, flags: int, lun: int) -> int:
        tag = self._next_tag()
        cdb_padded = cdb.ljust(16, b"\x00")
        cbw = struct.pack(
            "<I I I B B B",
            CBW_SIGNATURE,
            tag,
            data_len,
            flags,
            lun,
            len(cdb),
        ) + cdb_padded
        self.dev.write(self.ep_out, cbw)
        return tag

    def _read_csw(self, expected_tag: int) -> None:
        csw = bytes(self.dev.read(self.ep_in, 13))
        signature, tag, residue, status = struct.unpack("<I I I B", csw[:13])
        if signature != CSW_SIGNATURE or tag != expected_tag:
            raise RuntimeError("Invalid CSW received from device")
        if status != 0:
            raise RuntimeError(f"Command failed with status {status} (residue {residue})")

    def command(self, cdb: bytes, data_in_len: int = 0, data_out: bytes | None = None) -> bytes:
        flags = 0x80 if data_in_len > 0 else 0x00
        if data_out is not None:
            data_len = len(data_out)
            flags = 0x00
        else:
            data_len = data_in_len
        tag = self._send_cbw(cdb, data_len, flags, 0)
        data_in = b""
        if data_out is not None:
            self.dev.write(self.ep_out, data_out)
        elif data_in_len > 0:
            data_in = self._read_bulk(data_in_len)
        self._read_csw(tag)
        return data_in

    def _read_bulk(self, length: int) -> bytes:
        data = bytearray()
        remaining = length
        while remaining > 0:
            chunk = bytes(self.dev.read(self.ep_in, remaining))
            data.extend(chunk)
            if len(chunk) < remaining:
                break
            remaining -= len(chunk)
        return bytes(data)

    def close(self) -> None:
        usb.util.release_interface(self.dev, DEFAULT_INTERFACE)
        usb.util.dispose_resources(self.dev)


def wait_ready(ms: UsbMassStorage, retries: int = 100, delay_s: float = 0.1) -> None:
    cdb = bytes([0x00, 0x00, 0x00, 0x00, 0x00, 0x00])  # TEST UNIT READY
    for _ in range(retries):
        try:
            ms.command(cdb)
            return
        except Exception:
            time.sleep(delay_s)
    raise RuntimeError("Camera did not become ready in time")


def scsi_inquiry(ms: UsbMassStorage) -> tuple[str, str, str]:
    cdb = bytes([0x12, 0x00, 0x00, 0x00, 0x24, 0x00])  # INQUIRY, alloc 36
    data = ms.command(cdb, data_in_len=0x24)
    vendor = data[8:16].decode("ascii", errors="ignore").strip()
    product = data[16:32].decode("ascii", errors="ignore").strip()
    revision = data[32:36].decode("ascii", errors="ignore").strip()
    return vendor, product, revision


def download_data(ms: UsbMassStorage) -> bytes:
    ms.command(bytes([0xC6, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]), data_in_len=65536)
    result = bytearray()
    packet = 0
    while True:
        cdb = bytes([0xC4, 0x00, 0x01, 0x00, 0x00, packet & 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        chunk = ms.command(cdb, data_in_len=65536)
        result.extend(chunk)
        if len(chunk) < 65536:
            break
        packet += 1
    return bytes(result)


def get_camera_information(ms: UsbMassStorage) -> dict:
    vendor, product, revision = scsi_inquiry(ms)
    ms.command(bytes([0xC2, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]))
    info_response = download_data(ms)
    serial = info_response[0x0100:].split(b"\x00", 1)[0].decode("ascii", errors="ignore").strip()
    firmware = info_response[0x0200:].split(b"\x00", 1)[0].decode("ascii", errors="ignore").strip()
    return {
        "vendor": vendor,
        "product": product,
        "revision": revision,
        "serial": serial,
        "firmware": firmware,
    }


def main() -> int:
    dev = usb.core.find(idVendor=LYTRO_VENDOR_ID, idProduct=LYTRO_PRODUCT_ID)
    if dev is None:
        print("Lytro camera not found.")
        return 1

    ms = UsbMassStorage(dev)
    try:
        wait_ready(ms)
        info = get_camera_information(ms)
    finally:
        ms.close()

    print(f"Vendor: {info['vendor']}")
    print(f"Product: {info['product']}")
    print(f"Revision: {info['revision']}")
    print(f"Serial: {info['serial']}")
    print(f"Firmware: {info['firmware']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
