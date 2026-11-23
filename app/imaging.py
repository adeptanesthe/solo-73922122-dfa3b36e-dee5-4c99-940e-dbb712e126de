"""PNG decoding and validation helpers."""

from __future__ import annotations

import io
import json
import math
import re
from dataclasses import dataclass
from typing import Any

from PIL import Image, UnidentifiedImageError

RGB = tuple[int, int, int]

MAX_PAGES = 4
MAX_SIDE = 128
MAX_LABELS = 12
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MIN_PALETTE_COLORS = 2
MAX_PALETTE_COLORS = 6

_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{6})$")


class InputError(ValueError):
    """Raised when submitted data does not satisfy the input contract."""


@dataclass(frozen=True)
class Page:
    index: int
    width: int
    height: int
    rgb: bytes  # raw RGB bytes, row-major
    labels: bytes  # raw 8-bit grayscale bytes, row-major
    rgb_name: str = ""
    label_name: str = ""


def _open_png(data: bytes, filename: str) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(data))
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InputError(f"{filename}: not a readable image") from exc
    if image.format != "PNG":
        raise InputError(f"{filename}: expected a PNG image, got {image.format}")
    width, height = image.size
    if width < 1 or height < 1:
        raise InputError(f"{filename}: image is empty")
    if width > MAX_SIDE or height > MAX_SIDE:
        raise InputError(
            f"{filename}: sides must be at most {MAX_SIDE} pixels, got {width}x{height}"
        )
    try:
        image.load()
    except (OSError, ValueError) as exc:
        raise InputError(f"{filename}: corrupt PNG data") from exc
    return image


def load_page(
    rgb_data: bytes, label_data: bytes, index: int, rgb_name: str, label_name: str
) -> Page:
    rgb_image = _open_png(rgb_data, rgb_name)
    if rgb_image.mode != "RGB":
        raise InputError(
            f"{rgb_name}: page originals must be 8-bit RGB, got mode {rgb_image.mode}"
        )
    label_image = _open_png(label_data, label_name)
    if label_image.mode != "L":
        raise InputError(
            f"{label_name}: label maps must be 8-bit grayscale, got mode {label_image.mode}"
        )
    if rgb_image.size != label_image.size:
        raise InputError(
            f"{label_name}: size {label_image.width}x{label_image.height} does not match "
            f"{rgb_name} size {rgb_image.width}x{rgb_image.height}"
        )
    width, height = rgb_image.size
    return Page(
        index=index,
        width=width,
        height=height,
        rgb=rgb_image.tobytes(),
        labels=label_image.tobytes(),
        rgb_name=rgb_name,
        label_name=label_name,
    )


def parse_palette(raw: str) -> list[RGB]:
    try:
        value: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InputError("palette must be a JSON array of colours") from exc
    if not isinstance(value, list):
        raise InputError("palette must be a JSON array of colours")
    if not MIN_PALETTE_COLORS <= len(value) <= MAX_PALETTE_COLORS:
        raise InputError(
            f"palette must contain {MIN_PALETTE_COLORS} to {MAX_PALETTE_COLORS} colours"
        )
    colours: list[RGB] = []
    for item in value:
        colours.append(parse_colour(item, "palette colour"))
    if len(set(colours)) != len(colours):
        raise InputError("palette colours must be distinct")
    return colours


def parse_colour(value: Any, what: str = "colour") -> RGB:
    """Accept ``#rrggbb`` / ``rrggbb`` strings or ``[r, g, b]`` arrays."""
    if isinstance(value, str):
        match = _HEX_RE.match(value.strip())
        if not match:
            raise InputError(f"{what} must look like #rrggbb, got {value!r}")
        digits = match.group(1)
        return (int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16))
    if isinstance(value, (list, tuple)) and len(value) == 3:
        channels: list[int] = []
        for channel in value:
            if isinstance(channel, bool) or not isinstance(channel, int):
                raise InputError(f"{what} channels must be integers")
            if not 0 <= channel <= 255:
                raise InputError(f"{what} channels must be between 0 and 255")
            channels.append(channel)
        return (channels[0], channels[1], channels[2])
    raise InputError(f"{what} must be #rrggbb or [r, g, b]")


def parse_min_distance(value: float) -> float:
    try:
        distance = float(value)
    except (TypeError, ValueError) as exc:
        raise InputError("min_distance must be a number") from exc
    if not math.isfinite(distance):
        raise InputError("min_distance must be finite")
    if distance <= 0:
        raise InputError("min_distance must be greater than zero")
    return distance


def parse_lock_colour(value: Any) -> RGB:
    if isinstance(value, dict) and "rgb" in value:
        value = value["rgb"]
    return parse_colour(value, "lock colour")


def colour_to_hex(colour: RGB) -> str:
    return "#{:02x}{:02x}{:02x}".format(*colour)
