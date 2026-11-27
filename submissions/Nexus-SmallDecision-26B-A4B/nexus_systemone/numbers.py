"""Number normalization: thousands separators, currency magnitudes, unit
spacing, and named-color annotations for hex colors."""

from __future__ import annotations

import re
from decimal import Decimal

_THOUSANDS = re.compile(r"(?<![\d.,])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d,]|\.\d)")
_CURRENCY_MAGNITUDE = re.compile(
    r"([$€£¥])\s?(\d+(?:\.\d+)?)\s?(k|K|m|M|mm|MM|bn|b|B)\b"
)
_MAGNITUDES = {
    "k": 10**3,
    "K": 10**3,
    "m": 10**6,
    "M": 10**6,
    "mm": 10**6,
    "MM": 10**6,
    "bn": 10**9,
    "b": 10**9,
    "B": 10**9,
}
# Only units whose attached form is unambiguous: "5m" (meters, minutes,
# million) and "5in" are deliberately absent.
_UNITS = (
    "km",
    "cm",
    "mm",
    "kg",
    "mg",
    "ml",
    "mph",
    "kph",
    "kWh",
    "GHz",
    "MHz",
    "KB",
    "MB",
    "GB",
    "TB",
    "ms",
    "hrs",
    "hr",
    "min",
    "lbs",
    "lb",
    "oz",
)
_UNIT_SPACING = re.compile(rf"(?<![\w.])(\d+(?:\.\d+)?)({'|'.join(_UNITS)})\b")

_NAMED_COLORS = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "green": (0, 128, 0),
    "lime": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "gray": (128, 128, 128),
    "silver": (192, 192, 192),
    "maroon": (128, 0, 0),
    "olive": (128, 128, 0),
    "purple": (128, 0, 128),
    "teal": (0, 128, 128),
    "navy": (0, 0, 128),
    "orange": (255, 165, 0),
    "pink": (255, 192, 203),
    "brown": (165, 42, 42),
    "gold": (255, 215, 0),
    "beige": (245, 245, 220),
    "violet": (238, 130, 238),
    "indigo": (75, 0, 130),
    "coral": (255, 127, 80),
    "salmon": (250, 128, 114),
    "khaki": (240, 230, 140),
    "turquoise": (64, 224, 208),
    "lavender": (230, 230, 250),
    "crimson": (220, 20, 60),
    "tan": (210, 180, 140),
}
# Six-digit hex always; three-digit only with a letter so "#123" issue numbers
# are left alone.
_HEX = re.compile(
    r"(?<![\w&])#([0-9a-fA-F]{6}|(?=[0-9]*[a-fA-F])[0-9a-fA-F]{3})\b(?!\s\(color:)"
)


def _strip_thousands(m: re.Match[str]) -> str:
    return m.group(0).replace(",", "")


def _expand_magnitude(m: re.Match[str]) -> str:
    value = Decimal(m.group(2)) * _MAGNITUDES[m.group(3)]
    return f"{m.group(1)}{value.normalize():f}"


def nearest_color(rgb: tuple[int, int, int]) -> tuple[str, bool]:
    """Nearest named color and whether the match is exact."""
    name, ref = min(
        _NAMED_COLORS.items(),
        key=lambda item: sum((a - b) ** 2 for a, b in zip(item[1], rgb)),
    )
    return name, ref == rgb


def _annotate_hex(m: re.Match[str]) -> str:
    digits = m.group(1)
    if len(digits) == 3:
        digits = "".join(c * 2 for c in digits)
    rgb = (int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16))
    name, exact = nearest_color(rgb)
    return f"{m.group(0)} (color: {name if exact else 'near ' + name})"


def normalize_numbers(text: str) -> str:
    text = _THOUSANDS.sub(_strip_thousands, text)
    text = _CURRENCY_MAGNITUDE.sub(_expand_magnitude, text)
    text = _UNIT_SPACING.sub(r"\1 \2", text)
    return _HEX.sub(_annotate_hex, text)
