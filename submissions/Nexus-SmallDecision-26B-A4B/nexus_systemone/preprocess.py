"""The deterministic preprocessor. It is versioned with the weights, like a
tokenizer: the model always trains and serves on its output."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .canonical import format_number, json_key_order
from .dates import normalize_dates
from .escape import SpecialTokenEscaper
from .numbers import normalize_numbers

PREPROCESSOR_VERSION = "1.0.0"

IMAGE_KEY = "x_nexus_image"
# Marks where an image sat in the state; this runtime is text-only, so the
# renderer refuses any state that contains one.
IMAGE_SENTINEL = "\x00IMAGE\x00"

_HSPACE = re.compile(r"[ \t\f\v\u00a0\u2000-\u200a\u202f\u205f\u3000]+")
_TRAILING = re.compile(r" +\n")
_BLANK_RUNS = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class ImageRef:
    base64: str
    mime: str


@dataclass
class Rendered:
    text: str
    images: list[ImageRef] = field(default_factory=list)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HSPACE.sub(" ", text)
    text = _TRAILING.sub("\n", text)
    return _BLANK_RUNS.sub("\n\n", text).strip()


class InvalidImage(ValueError):
    """An ``x_nexus_image`` object is malformed or does not decode."""


def _image_ref(value: Any) -> ImageRef:
    if not isinstance(value, dict) or not isinstance(value.get("base64"), str):
        raise InvalidImage(f"{IMAGE_KEY} must be an object with a base64 string")
    mime = value.get("mime", "image/png")
    if not isinstance(mime, str) or not mime.startswith("image/"):
        raise InvalidImage(f"{IMAGE_KEY}.mime must be an image MIME type, got {mime!r}")
    return ImageRef(base64=value["base64"], mime=mime)


class Preprocessor:
    """Text normalization plus the canonical renderer for structured values.

    ``specials`` are the special-token strings of the serving tokenizer
    (the model's reserved tokens plus the base model's own); any literal
    occurrence in caller text is escaped.
    """

    def __init__(self, specials: Iterable[str], now: str | None = None):
        self._escape = SpecialTokenEscaper(specials)
        self.now = now

    def text(self, value: str) -> str:
        value = unicodedata.normalize("NFC", value)
        value = normalize_whitespace(value)
        value = normalize_dates(value, self.now)
        value = normalize_numbers(value)
        return self._escape(value)

    def key(self, value: str) -> str:
        return self._escape(normalize_whitespace(unicodedata.normalize("NFC", value)))

    def render(self, value: Any) -> Rendered:
        """Render a state, instructions or criteria value.

        A top-level string renders bare. Structures render as canonical JSON
        (RFC 8785 key order and number format) with every array item prefixed
        by ``[i/n]``; ``x_nexus_image`` objects become image markers.
        """
        images: list[ImageRef] = []
        if isinstance(value, str):
            return Rendered(self.text(value), images)
        return Rendered(self._render(value, images), images)

    def _render(self, value: Any, images: list[ImageRef]) -> str:
        if isinstance(value, dict):
            if IMAGE_KEY in value:
                if len(value) != 1:
                    raise InvalidImage(f"an {IMAGE_KEY} object must have no other keys")
                images.append(_image_ref(value[IMAGE_KEY]))
                return IMAGE_SENTINEL
            parts = [
                f"{json.dumps(self.key(k), ensure_ascii=False)}: {self._render(value[k], images)}"
                for k in json_key_order(value)
            ]
            return "{" + ", ".join(parts) + "}"
        if isinstance(value, list):
            n = len(value)
            items = [
                f"[{i}/{n}] {self._render(v, images)}" for i, v in enumerate(value, 1)
            ]
            return "[" + ", ".join(items) + "]"
        if isinstance(value, str):
            return json.dumps(self.text(value), ensure_ascii=False)
        if isinstance(value, bool) or value is None:
            return json.dumps(value)
        if isinstance(value, (int, float)):
            return format_number(value)
        raise TypeError(f"unsupported JSON value of type {type(value).__name__}")
