from __future__ import annotations

from dataclasses import dataclass
import json


def _find_json_blob(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Metadata does not contain a JSON object")
    return text[start : end + 1]


@dataclass(frozen=True)
class RawDetails:
    right_shift: int
    black: dict[str, int]
    white: dict[str, int]
    endianness: str
    bits_per_pixel: int
    mosaic_tile: str
    mosaic_upper_left: str


@dataclass(frozen=True)
class ImageInfo:
    width: int
    height: int
    orientation: int
    representation: str
    raw: RawDetails


@dataclass(frozen=True)
class LensConfig:
    zoom_step: int
    focus_step: int


@dataclass(frozen=True)
class Metadata:
    raw_json: dict

    @classmethod
    def from_bytes(cls, data: bytes) -> "Metadata":
        text = data.decode("utf-8", errors="ignore")
        blob = _find_json_blob(text)
        return cls(raw_json=json.loads(blob))

    def image_info(self) -> ImageInfo:
        img = self.raw_json["master"]["picture"]["frameArray"][0]["frame"]["metadata"][
            "image"
        ]
        raw_details = img["rawDetails"]
        pixfmt = raw_details["pixelFormat"]
        packing = raw_details["pixelPacking"]
        mosaic = raw_details["mosaic"]
        raw = RawDetails(
            right_shift=int(pixfmt["rightShift"]),
            black={
                "r": int(pixfmt["black"]["r"]),
                "gr": int(pixfmt["black"]["gr"]),
                "gb": int(pixfmt["black"]["gb"]),
                "b": int(pixfmt["black"]["b"]),
            },
            white={
                "r": int(pixfmt["white"]["r"]),
                "gr": int(pixfmt["white"]["gr"]),
                "gb": int(pixfmt["white"]["gb"]),
                "b": int(pixfmt["white"]["b"]),
            },
            endianness=str(packing["endianness"]),
            bits_per_pixel=int(packing["bitsPerPixel"]),
            mosaic_tile=str(mosaic["tile"]),
            mosaic_upper_left=str(mosaic["upperLeftPixel"]),
        )
        return ImageInfo(
            width=int(img["width"]),
            height=int(img["height"]),
            orientation=int(img["orientation"]),
            representation=str(img["representation"]),
            raw=raw,
        )

    def lens_config(self) -> LensConfig:
        lens = self.raw_json["master"]["picture"]["frameArray"][0]["frame"]["metadata"][
            "devices"
        ]["lens"]
        return LensConfig(
            zoom_step=int(lens["zoomStep"]),
            focus_step=int(lens["focusStep"]),
        )
