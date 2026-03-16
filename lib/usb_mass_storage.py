import asyncio
import struct
import usb

DEFAULT_EP_OUT = 0x02
DEFAULT_EP_IN = 0x82
DEFAULT_INTERFACE = 0
DEFAULT_TIMEOUT_MS = 5000


CBW_SIGNATURE = 0x43425355  # "USBC"
CSW_SIGNATURE = 0x53425355  # "USBS"


class UsbMassStorage:
    """Generic USB Mass Storage (Bulk-Only Transport) implementation."""

    def __init__(self, dev: usb.core.Device) -> None:
        self.dev = dev
        self._detach_kernel_driver()
        try:
            self.dev.set_configuration()
        except usb.core.USBError as exc:
            if exc.errno == 16:
                raise RuntimeError(
                    "USB resource busy. Another driver/app is using the camera. "
                    "Try unmounting the volume or detaching the kernel "
                    "driver, then re-run."
                ) from exc
            raise
        usb.util.claim_interface(self.dev, DEFAULT_INTERFACE)
        self.ep_out, self.ep_in = self._find_endpoints()
        self._tag = 1
        self.timeout_ms = DEFAULT_TIMEOUT_MS

    def _detach_kernel_driver(self) -> None:
        try:
            if self.dev.is_kernel_driver_active(DEFAULT_INTERFACE):
                self.dev.detach_kernel_driver(DEFAULT_INTERFACE)
        except NotImplementedError:
            return

    def _find_endpoints(self) -> tuple[int, int]:
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
        return ep_out.bEndpointAddress, ep_in.bEndpointAddress  # type: ignore[attr-defined]

    def _next_tag(self) -> int:
        tag = self._tag
        self._tag = (self._tag + 1) & 0xFFFFFFFF
        if self._tag == 0:
            self._tag = 1
        return tag

    def _send_cbw(
        self, cdb: bytes, data_len: int, flags: int, lun: int, timeout_ms: int
    ) -> int:
        tag = self._next_tag()
        cdb_padded = cdb.ljust(16, b"\x00")
        cbw = (
            struct.pack(
                "<I I I B B B",
                CBW_SIGNATURE,
                tag,
                data_len,
                flags,
                lun,
                len(cdb),
            )
            + cdb_padded
        )
        self.dev.write(self.ep_out, cbw, timeout=timeout_ms)
        return tag

    def _read_csw(self, expected_tag: int, timeout_ms: int) -> None:
        csw = bytes(self.dev.read(self.ep_in, 13, timeout=timeout_ms))
        signature, tag, residue, status = struct.unpack("<I I I B", csw[:13])
        if signature != CSW_SIGNATURE or tag != expected_tag:
            raise RuntimeError("Invalid CSW received from device")
        if status != 0:
            raise RuntimeError(
                f"Command failed with status {status} (residue {residue})"
            )

    def _read_bulk(self, length: int, timeout_ms: int) -> bytes:
        data = bytearray()
        remaining = length
        while remaining > 0:
            chunk = bytes(self.dev.read(self.ep_in, remaining, timeout=timeout_ms))
            data.extend(chunk)
            if len(chunk) < remaining:
                break
            remaining -= len(chunk)
        return bytes(data)

    def _command_sync(
        self,
        cdb: bytes,
        data_in_len: int = 0,
        data_out: bytes | None = None,
        timeout_ms: int | None = None,
    ) -> bytes:
        """Synchronous backend for executing commands to avoid blocking the async event loop."""
        effective_timeout = self.timeout_ms if timeout_ms is None else timeout_ms
        flags = 0x80 if data_in_len > 0 else 0x00
        if data_out is not None:
            data_len = len(data_out)
            flags = 0x00
        else:
            data_len = data_in_len

        tag = self._send_cbw(cdb, data_len, flags, 0, effective_timeout)
        data_in = b""

        if data_out is not None:
            self.dev.write(self.ep_out, data_out, timeout=effective_timeout)
        elif data_in_len > 0:
            data_in = self._read_bulk(data_in_len, effective_timeout)

        self._read_csw(tag, effective_timeout)
        return data_in

    async def command(
        self,
        cdb: bytes,
        data_in_len: int = 0,
        data_out: bytes | None = None,
        timeout_ms: int | None = None,
    ) -> bytes:
        """Async public interface for sending SCSI commands."""
        return await asyncio.to_thread(
            self._command_sync, cdb, data_in_len, data_out, timeout_ms
        )

    def close(self) -> None:
        usb.util.release_interface(self.dev, DEFAULT_INTERFACE)
        usb.util.dispose_resources(self.dev)
