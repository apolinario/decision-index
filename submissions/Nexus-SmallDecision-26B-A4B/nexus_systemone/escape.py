"""Neutralize literal special-token strings in caller text so state content can
never forge a delimiter or an answer slot."""

from __future__ import annotations

import re
from collections.abc import Iterable

# Visually close stand-ins (‹ › ⟦ ⟧ ¦ ∕); none appear in any special token
# string. "/" covers MiniCPM-style control tokens such as "/no_think".
_LOOKALIKES = str.maketrans(
    {
        "<": "\u2039",
        ">": "\u203a",
        "[": "\u27e6",
        "]": "\u27e7",
        "|": "\u00a6",
        "/": "\u2215",
    }
)


class SpecialTokenEscaper:
    """Rewrites any occurrence of the given special-token strings."""

    def __init__(self, specials: Iterable[str]):
        tokens = sorted({t for t in specials if t}, key=len, reverse=True)
        for token in tokens:
            if token.translate(_LOOKALIKES) == token:
                raise ValueError(
                    f"special token {token!r} has no delimiter character to escape"
                )
        self._pattern = (
            re.compile("|".join(re.escape(t) for t in tokens)) if tokens else None
        )

    def __call__(self, text: str) -> str:
        if self._pattern is None:
            return text
        return self._pattern.sub(lambda m: m.group(0).translate(_LOOKALIKES), text)
