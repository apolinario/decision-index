"""The serving tokenizer: the release folder's ``tokenizer.json`` read with the
``tokenizers`` library, and the chat layout the model was trained on.

The layout is the release ``chat_template.jinja`` for the only message shapes
this runtime sends (system and user turns, thinking off), written out as plain
string joins. ``CHAT_TEMPLATE_SHA256`` pins the template file it was checked
against, so a tokenizer folder with another template is refused instead of
silently rendering a different prompt.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path

from tokenizers import Tokenizer

from .tokens import RESERVED_TOKENS

CHAT_TEMPLATE_SHA256 = (
    "94899c0f917d93f6fe81c95744d1e8ddab2d21d39228d2e4aec1fb2a25bff413"
)
BOS = "<bos>"
TURN_START = "<|turn>"
TURN_END = "<turn|>"
# The empty thought block the template writes when thinking is off.
GENERATION_PROMPT = "<|turn>model\n<|channel>thought\n<channel|>"
ROLES = ("system", "user")


def apply_chat_template(
    messages: Sequence[dict[str, str]], add_generation_prompt: bool
) -> str:
    """The release template's text for system and user messages. Content is
    stripped, as the template's ``trim`` does."""
    parts = [BOS]
    for message in messages:
        role = message["role"]
        if role not in ROLES:
            raise ValueError(
                f"the chat layout renders {ROLES} messages only, got role {role!r}"
            )
        parts.append(f"{TURN_START}{role}\n{message['content'].strip()}{TURN_END}\n")
    if add_generation_prompt:
        parts.append(GENERATION_PROMPT)
    return "".join(parts)


class ServingTokenizer:
    """Encodes prompt text and knows which strings the preprocessor must
    escape. ``folder`` is a copy of the model repo's tokenizer files."""

    def __init__(self, folder: Path | str):
        folder = Path(folder)
        tokenizer_file = folder / "tokenizer.json"
        template_file = folder / "chat_template.jinja"
        for required in (tokenizer_file, template_file):
            if not required.is_file():
                raise FileNotFoundError(
                    f"{required} is missing; point --tokenizer at a copy of the "
                    "model repo's tokenizer files"
                )
        digest = hashlib.sha256(template_file.read_bytes()).hexdigest()
        if digest != CHAT_TEMPLATE_SHA256:
            raise ValueError(
                f"{template_file} has sha256 {digest}, not the release template's "
                f"{CHAT_TEMPLATE_SHA256}; this runtime renders only that template"
            )
        self._tokenizer = Tokenizer.from_file(str(tokenizer_file))
        added = json.loads(tokenizer_file.read_text(encoding="utf-8"))["added_tokens"]
        self._specials = sorted(
            {
                t["content"]
                for t in added
                if t["special"] or not re.fullmatch(r"\w+", t["content"])
            }
            | set(RESERVED_TOKENS)
        )
        encoded = {t: self.encode(t) for t in RESERVED_TOKENS}
        split = [t for t, ids in encoded.items() if len(ids) != 1]
        if split:
            raise ValueError(
                f"{tokenizer_file} does not hold the model's reserved tokens as "
                f"single tokens (first missing: {split[:3]}); it is not the "
                "release tokenizer"
            )
        if len({ids[0] for ids in encoded.values()}) != len(encoded):
            raise ValueError(
                f"{tokenizer_file} gives several reserved tokens the same id "
                "(an unknown-token collapse); it is not the release tokenizer"
            )

    def encode(self, text: str) -> list[int]:
        """Token ids without any added start token: the rendered text carries
        its own ``<bos>``."""
        return self._tokenizer.encode(text, add_special_tokens=False).ids

    def special_strings(self) -> list[str]:
        return list(self._specials)

    def render_chat(
        self, messages: Sequence[dict[str, str]], add_generation_prompt: bool
    ) -> str:
        return apply_chat_template(messages, add_generation_prompt)
