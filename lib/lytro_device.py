import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TypedDict
import re

import usb

from lib.usb_mass_storage import UsbMassStorage

LYTRO_VENDOR_ID = 0x24CF
LYTRO_PRODUCT_ID = 0x00A1


@dataclass(frozen=True)
class CameraInfo:
    vendor: str
    product: str
    revision: str
    serial: str
    firmware: str


@dataclass(frozen=True)
class PictureEntry:
    dir_base: str
    file_base: str
    dir_id: int
    file_id: int
    sha1_hex: str
    captured_at: datetime | None
    path: str
    basename: str

    @property
    def full_path(self) -> str:
        return f"{self.path}{self.basename}"

    @property
    def metadata_path(self) -> str:
        return f"{self.full_path}.TXT"

    @property
    def raw_path(self) -> str:
        return f"{self.full_path}.RAW"

    @property
    def thumbnail_path(self) -> str:
        return f"{self.full_path}.128"


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

    @staticmethod
    def _decode_c_string(raw: bytes) -> str:
        return raw.split(b"\x00", 1)[0].decode("ascii", errors="ignore").strip()

    @staticmethod
    def _build_picture_path(dir_base: str, dir_id: int) -> str:
        return f"I:\\DCIM\\{dir_id}{dir_base}\\"

    @staticmethod
    def _build_picture_basename(file_base: str, file_id: int) -> str:
        return f"{file_base}{file_id:04d}"

    @staticmethod
    def _parse_sha1_hex(raw: bytes) -> str:
        candidate = raw.decode("ascii", errors="ignore")
        cleaned = "".join(ch for ch in candidate if ch in "0123456789abcdefABCDEF")
        return cleaned[:40].lower()

    @staticmethod
    def _parse_picture_list(data: bytes) -> list[PictureEntry]:
        if len(data) < 12:
            return []
        line_len = int.from_bytes(data[4:8], "little", signed=False)
        entry_offset = int.from_bytes(data[8:12], "little", signed=False)
        if line_len <= 0:
            return []

        def parse_at(pos: int) -> list[PictureEntry]:
            entries: list[PictureEntry] = []
            while pos + line_len <= len(data):
                line = data[pos : pos + line_len]
                dir_base = LytroDevice._decode_c_string(line[0:8])
                file_base = LytroDevice._decode_c_string(line[8:16])
                dir_id = int.from_bytes(line[16:20], "little", signed=False)
                file_id = int.from_bytes(line[20:24], "little", signed=False)
                sha1_hex = LytroDevice._parse_sha1_hex(line[53:93])

                captured_at = None
                time_raw = LytroDevice._decode_c_string(line[96:120])
                if time_raw:
                    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
                        try:
                            captured_at = datetime.strptime(time_raw, fmt).replace(
                                tzinfo=timezone.utc
                            )
                            break
                        except ValueError:
                            continue

                path = LytroDevice._build_picture_path(dir_base, dir_id)
                basename = LytroDevice._build_picture_basename(file_base, file_id)
                entries.append(
                    PictureEntry(
                        dir_base=dir_base,
                        file_base=file_base,
                        dir_id=dir_id,
                        file_id=file_id,
                        sha1_hex=sha1_hex,
                        captured_at=captured_at,
                        path=path,
                        basename=basename,
                    )
                )
                pos += line_len
            return entries

        dir_pattern = re.compile(r"^[0-9]{3}PHOTO$", re.IGNORECASE)
        file_pattern = re.compile(r"^(IMG_|MOD_)", re.IGNORECASE)

        def is_valid_entry(entry: PictureEntry) -> bool:
            if entry.dir_id <= 0 or entry.file_id < 0:
                return False
            if not entry.dir_base or not entry.file_base:
                return False
            if dir_pattern.match(entry.dir_base) is None:
                return False
            if file_pattern.match(entry.file_base) is None:
                return False
            return True

        # Some devices appear to store entry_offset in bytes instead of 8-byte units.
        pos_a = entry_offset * 8 + 12
        pos_b = entry_offset + 12
        entries_a = parse_at(pos_a) if pos_a < len(data) else []
        entries_b = parse_at(pos_b) if pos_b < len(data) else []
        candidates = [entries_a, entries_b]

        # Heuristic alignment scan: try all offsets within a line to find the most valid entries.
        best_entries = entries_a if len(entries_a) >= len(entries_b) else entries_b
        best_valid = sum(1 for e in best_entries if is_valid_entry(e))
        for offset in range(line_len):
            pos = 12 + offset
            if pos >= len(data):
                continue
            trial = parse_at(pos)
            valid = sum(1 for e in trial if is_valid_entry(e))
            if valid > best_valid:
                best_valid = valid
                best_entries = trial
        # Filter to valid entries only; if that empties the list, fall back to raw.
        filtered = [e for e in best_entries if is_valid_entry(e)]
        return filtered if filtered else best_entries

    @staticmethod
    def _debug_picture_list(data: bytes) -> dict:
        if len(data) < 12:
            return {"error": "data too short"}
        line_len = int.from_bytes(data[4:8], "little", signed=False)
        entry_offset = int.from_bytes(data[8:12], "little", signed=False)
        if line_len <= 0:
            return {"error": "invalid line_len"}

        dir_pattern = re.compile(r"^[0-9]{3}PHOTO$", re.IGNORECASE)
        file_pattern = re.compile(r"^(IMG_|MOD_)", re.IGNORECASE)

        def is_valid_entry(entry: PictureEntry) -> bool:
            if entry.dir_id <= 0 or entry.file_id < 0:
                return False
            if not entry.dir_base or not entry.file_base:
                return False
            if dir_pattern.match(entry.dir_base) is None:
                return False
            if file_pattern.match(entry.file_base) is None:
                return False
            return True

        def parse_at(pos: int) -> list[PictureEntry]:
            entries: list[PictureEntry] = []
            while pos + line_len <= len(data):
                line = data[pos : pos + line_len]
                dir_base = LytroDevice._decode_c_string(line[0:8])
                file_base = LytroDevice._decode_c_string(line[8:16])
                dir_id = int.from_bytes(line[16:20], "little", signed=False)
                file_id = int.from_bytes(line[20:24], "little", signed=False)
                sha1_hex = LytroDevice._parse_sha1_hex(line[53:93])
                path = LytroDevice._build_picture_path(dir_base, dir_id)
                basename = LytroDevice._build_picture_basename(file_base, file_id)
                entries.append(
                    PictureEntry(
                        dir_base=dir_base,
                        file_base=file_base,
                        dir_id=dir_id,
                        file_id=file_id,
                        sha1_hex=sha1_hex,
                        captured_at=None,
                        path=path,
                        basename=basename,
                    )
                )
                pos += line_len
            return entries

        pos_a = entry_offset * 8 + 12
        pos_b = entry_offset + 12
        entries_a = parse_at(pos_a) if pos_a < len(data) else []
        entries_b = parse_at(pos_b) if pos_b < len(data) else []
        scan = []
        for offset in range(min(line_len, 256)):
            pos = 12 + offset
            if pos >= len(data):
                break
            trial = parse_at(pos)
            valid = sum(1 for e in trial if is_valid_entry(e))
            scan.append((offset, len(trial), valid))
        scan_sorted = sorted(scan, key=lambda x: (x[2], x[1]), reverse=True)[:5]
        return {
            "data_len": len(data),
            "line_len": line_len,
            "entry_offset": entry_offset,
            "pos_a": pos_a,
            "pos_b": pos_b,
            "entries_a": len(entries_a),
            "entries_b": len(entries_b),
            "top_offsets": scan_sorted,
        }

    async def get_picture_list_raw(self, timeout_ms: int | None = 15000) -> bytes:
        await self.command(
            bytes(
                [0xC2, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
            ),
            timeout_ms=timeout_ms,
        )
        return await self.download_data()

    async def get_picture_list(self) -> list[PictureEntry]:
        raw = await self.get_picture_list_raw()
        return self._parse_picture_list(raw)

    async def get_file(self, file_path: str) -> bytes:
        file_bytes = file_path.encode("ascii", errors="ignore") + b"\x00"
        await self.command(
            bytes(
                [0xC2, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
            ),
            data_out=file_bytes,
        )
        return await self.download_data()

    async def get_file_text(self, file_path: str) -> str:
        data = await self.get_file(file_path)
        return data.decode("utf-8", errors="ignore")

    async def get_firmware_text(self) -> str:
        return await self.get_file_text("A:\\FIRMWARE.TXT")

    async def get_vcm_text(self) -> str:
        return await self.get_file_text("A:\\VCM.TXT")

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
        return CameraInfo(
            vendor=vendor,
            product=product,
            revision=revision,
            serial=serial,
            firmware=firmware,
        )
