"""Render each benchmark question in the checkpoint's input format."""

import json
import re
from string import ascii_uppercase

LETTERS = ascii_uppercase
MAX_OPTIONS = 255
SLOT = "\x00IMAGE\x00"
IMAGE_MARK = "<image>"
IMAGE_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"
SYSTEM = "Apply the criterion to the evidence. Choose exactly one listed option. Answer with its letter only."
THINK_OFF_SUFFIX = "<think>\n\n</think>\n\n"
BARE_OPTION_ID = re.compile(r"\(?[a-z][\).]?|\(?\d{1,3}[\).]?|opt(?:ion)?[_ ]?\d{1,3}", re.I)


def indexed_evidence(value):
    if isinstance(value, dict):
        return {key: indexed_evidence(item) for key, item in value.items()}
    if isinstance(value, list):
        items = [indexed_evidence(item) for item in value]
        if len(items) >= 8:
            return [
                {"_index": index, **(item if isinstance(item, dict) else {"value": item})}
                for index, item in enumerate(items)
            ]
        return items
    return value


def as_text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def description(value):
    return None if value is None else as_text(value).strip() or None


def evidence_text(state, image_count):
    text = "" if state in (None, {}, []) else as_text(indexed_evidence(state))
    if not image_count:
        return text
    marks = text.count(IMAGE_MARK)
    if marks:
        if marks != image_count:
            raise ValueError(f"found {marks} image positions for {image_count} images")
        return text.replace(IMAGE_MARK, SLOT)
    labels = [f"Image {number}: {SLOT}" for number in range(1, image_count + 1)]
    prefix = SLOT if image_count == 1 else "\n".join(labels)
    return prefix + (f"\n\n{text}" if text.strip() else "")


def render_question(state, spec, image_count=0):
    kind = spec.get("type", "choice")
    criteria = spec.get("criteria", spec.get("options"))
    question = description(spec.get("instructions", spec.get("question"))) or ""
    if kind == "choice":
        if isinstance(criteria, (list, tuple)):
            criteria = dict.fromkeys(map(str, criteria))
        if not isinstance(criteria, dict):
            raise ValueError("choice criteria must be a mapping or a list of options")
        choices = criteria.items()
    elif kind == "score":
        if isinstance(criteria, dict):
            try:
                levels = sorted(criteria, key=float)
            except (ValueError, TypeError):
                levels = list(criteria)
            criteria = [criteria[level] for level in levels]
        if not isinstance(criteria, (list, tuple)):
            raise ValueError("score criteria must be ordered levels")
        choices = enumerate(criteria)
    elif kind in ("noul", "bool"):
        kind = "noul"
        criteria = criteria if isinstance(criteria, dict) else {}
        choices = [(str(flag).lower(), criteria.get(str(flag).lower(), criteria.get(flag)))
                   for flag in (True, False)]
        question = question or "Which answer fits the evidence?"
    else:
        raise ValueError(f"unknown question type {kind!r}")
    return {
        "type": kind,
        "state": evidence_text(state, image_count),
        "instructions": question,
        "options": [{"id": str(key), "criterion": description(value)} for key, value in choices],
    }


def normalize_label(text):
    return re.sub(r"[\s_-]+", " ", text).strip().lower()


def messages(row, codes=LETTERS):
    options = row["options"]
    keys = [option["id"] for option in options]
    kind = row["type"]
    limit = {"choice": min(len(codes), MAX_OPTIONS), "score": 10, "noul": 2}[kind]
    question = row["instructions"].strip()
    if not question or not 2 <= len(keys) <= limit:
        raise ValueError(f"{kind} requires instructions and 2–{limit} options")
    if len(set(keys)) != len(keys) or any(
        not key.strip() or "\n" in key or "\r" in key or len(key) > 200
        for key in keys
    ):
        raise ValueError("option IDs must be distinct, nonempty single lines of at most 200 characters")
    hide_labels = kind == "choice" and all(
        BARE_OPTION_ID.fullmatch(option["id"]) and option["criterion"] for option in options
    )
    lines = []
    for code, option in zip(codes, options):
        label = option["id"]
        if kind == "noul":
            label = {"true": "yes", "false": "no"}[label]
        detail = option["criterion"]
        body = label
        if hide_labels:
            body = detail
        elif detail and normalize_label(detail) != normalize_label(label):
            body = f"{label}: {detail}"
        lines.append(f"{code}) {body}")
    state = row["state"]
    evidence = state if state.strip() else "(none)"
    body = f"Evidence:\n{evidence}\n\nQuestion: {question}\nOptions:\n" + "\n".join(lines)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}]


def content_parts(content):
    parts = []
    for index, part in enumerate(content.split(SLOT)):
        if index:
            parts.append({"type": "image"})
        if part:
            parts.append({"type": "text", "text": part})
    return parts


def plain_prompt(tokenizer, chat):
    image_count = sum(message["content"].count(SLOT) for message in chat)
    if image_count:
        chat = [{"role": message["role"], "content": content_parts(message["content"])} for message in chat]
    text = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    if image_count and text.count(IMAGE_TOKEN) != image_count:
        raise ValueError("chat template image count differs from the request")
    return text


def check_tokenizer(tokenizer):
    probe = render_question("s", {"type": "noul", "instructions": "q"})
    rendered = plain_prompt(tokenizer, messages(probe))
    if not rendered.endswith(THINK_OFF_SUFFIX):
        raise ValueError("chat template must end in the empty thinking block")
    prefix = tokenizer.encode(rendered, add_special_tokens=False)
    ids = []
    for letter in LETTERS:
        token = tokenizer.encode(letter, add_special_tokens=False)
        combined = tokenizer.encode(rendered + letter, add_special_tokens=False)
        if len(token) != 1 or combined != prefix + token:
            raise ValueError(f"letter {letter} must encode as a single token after the prompt")
        ids.extend(token)
    if len(set(ids)) != len(LETTERS):
        raise ValueError("letter token IDs must be distinct")
    return ids
