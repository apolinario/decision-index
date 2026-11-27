"""Canonical orderings and bytes, so client ordering can never change an answer."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import rfc8785


def option_order(keys: Iterable[str]) -> list[str]:
    """Choice option keys in bytewise UTF-8 order; this order assigns ``<opt_i>``."""
    return sorted(keys, key=lambda k: k.encode("utf-8"))


def json_key_order(keys: Iterable[str]) -> list[str]:
    """Object keys in RFC 8785 order (UTF-16 code units)."""
    return sorted(keys, key=lambda k: k.encode("utf-16-be"))


def canonical_bytes(value: Any) -> bytes:
    """RFC 8785 serialization, used for fingerprints and dedup keys."""
    return rfc8785.dumps(value)


def format_number(value: int | float) -> str:
    """ECMAScript number formatting as RFC 8785 prescribes."""
    return rfc8785.dumps(value).decode("ascii")
