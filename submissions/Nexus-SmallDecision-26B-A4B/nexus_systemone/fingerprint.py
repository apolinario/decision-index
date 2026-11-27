"""Serving fingerprint: one id that changes whenever anything that can change
an answer changes (prompt layout, preprocessor, chat template, served model,
context window, calibration, runtime)."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from . import RUNTIME_VERSION
from .canonical import canonical_bytes
from .preprocess import PREPROCESSOR_VERSION
from .tokenizer import CHAT_TEMPLATE_SHA256

PREFIX = "sd_"


def fingerprint_of(components: Mapping[str, Any]) -> str:
    """``sd_`` plus the first 16 hex digits of SHA-256 over the RFC 8785 form of
    ``components``; key order never matters."""
    digest = hashlib.sha256(canonical_bytes(dict(components))).hexdigest()
    return PREFIX + digest[:16]


def serving_components(
    *,
    backend_name: str,
    served_model: str,
    max_context: int,
    layout: str,
    temperatures: Mapping[str, float],
) -> dict[str, Any]:
    return {
        "runtime": RUNTIME_VERSION,
        "preprocessor": PREPROCESSOR_VERSION,
        "chat_template_sha256": CHAT_TEMPLATE_SHA256,
        "backend": backend_name,
        "served_model": served_model,
        "max_context": max_context,
        "layout": layout,
        "temperatures": dict(temperatures),
    }
