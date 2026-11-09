"""GLiClass engine: runs GLiClass zero-shot classifiers (https://github.com/Knowledgator/GLiClass) in process.

    pip install -e ".[gliclass]"
    python -m decision_index pipeline --engine gliclass --model knowledgator/gliclass-v4.1-edge \
        --option max_length=16384 --option batch_tokens=32768 --out runs/gliclass-v4.1-edge

Each question becomes one GLiClass classification:
  text   = the rendered state (a string as-is; a JSON object as `key: value` lines, list values as `- item` lines)
  prompt = the question instructions
  labels = the option descriptions (the option key when a description is None; duplicates get a `key: ` prefix)
A `noul` question uses its true/false criteria as labels ("yes" / "no" when it has none); its answer is p(true).
All questions of one request go through the model in one batch, split only to stay under batch_tokens.
Inputs longer than max_length tokens are refused as unsupported, never truncated.

Options (--option key=value): device, dtype (default bfloat16 on CUDA), max_length, batch_size (default 32),
batch_tokens (padded tokens per forward pass, default 65536),
probs = normalize (sigmoid scores / sum, default) | softmax (softmax over logits).
Set USE_FLASHDEBERTA=1 to use the FlashDeBERTa backend for DeBERTa encoders (same results, faster).
"""

import json
import math
import os

from decision_index.engines.base import Engine, Unsupported, text

NOUL_CRITERIA = {"true": "yes", "false": "no"}


def render_state(state):
    """Same rendering as the GLiClass training data."""
    if state in ("", None, {}, []):
        return ""
    if isinstance(state, str):
        return state
    if isinstance(state, list):
        return "\n".join(f"- {text(item)}" for item in state)
    lines = []
    for key, value in state.items():
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            lines.append(f"{key}:")
            lines.extend(f"- {item}" for item in value)
        else:
            lines.append(f"{key}: {text(value)}")
    return "\n".join(lines)


def option_labels(criteria):
    """Option key -> label text. Duplicate descriptions get the key as prefix so labels stay unique."""
    labels = {k: (k if d is None else text(d)) for k, d in criteria.items()}
    if len(set(labels.values())) != len(labels):
        labels = {k: f"{k}: {v}" if v != k else k for k, v in labels.items()}
    return labels


class GLiClassEngine(Engine):
    name = "gliclass"
    latency = "Device-synchronized in-process request wall time including rendering and tokenization; all questions of a request in one batch; excludes model loading."

    def __init__(self, model="knowledgator/gliclass-v4.1-edge", device=None, dtype=None, max_length=None,
                 batch_size=32, batch_tokens=65536, probs="normalize", **options):
        super().__init__(**options)
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        import torch
        from transformers import AutoTokenizer

        from gliclass import GLiClassModel, ZeroShotClassificationPipeline

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = dtype or ("bfloat16" if self.device.startswith("cuda") else "float32")
        self.model = GLiClassModel.from_pretrained(model).to(dtype=getattr(torch, dtype)).eval()
        self.tok = AutoTokenizer.from_pretrained(model)
        limit = getattr(self.model.config.encoder_config, "max_position_embeddings", None) if hasattr(self.model.config, "encoder_config") else None
        self.max_length = int(max_length or getattr(self.model.config, "max_length", None) or limit or 1024)
        self.pipe = ZeroShotClassificationPipeline(
            self.model, self.tok, max_classes=256, max_length=self.max_length,
            classification_type="multi-label", device=self.device, progress_bar=False,
        )
        if self.pipe.pipe.max_length < self.max_length:
            # the pipeline caps max_length to the encoder's position limit and would truncate silently
            raise ValueError(f"pipeline caps max_length to {self.pipe.pipe.max_length}, below the requested {self.max_length}")
        self.batch_size = int(batch_size)
        self.batch_tokens = int(batch_tokens)
        self.probs = probs
        self.model_id = model
        self.provenance = {
            "kind": "gliclass",
            "repo": model,
            "architecture_type": self.model.config.architecture_type,
            "device": self.device,
            "dtype": dtype,
            "max_length": self.max_length,
            "batch_tokens": self.batch_tokens,
            "flashdeberta": bool(os.environ.get("USE_FLASHDEBERTA")),
            "probs": probs,
            "policy": "text = rendered state, prompt = instructions, labels = option descriptions; noul labels = its true/false criteria (yes/no when empty). "
                      "Choice probabilities are the label scores normalized to sum 1 (or softmax over logits). "
                      "Inputs longer than max_length are refused as unsupported, never truncated.",
        }

    def runtime(self):
        info = {"torch": self.torch.__version__, "device": self.device}
        if self.device.startswith("cuda"):
            info.update(cuda=self.torch.version.cuda, gpu=self.torch.cuda.get_device_name())
        return info

    def synchronize(self):
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()

    def _n_tokens(self, body, prompt, labels):
        # labels + one label token each + sep + prompt + text + special tokens
        n = len(self.tok(prompt + body, add_special_tokens=True)["input_ids"]) + 1
        enc = self.tok(list(labels), add_special_tokens=False)["input_ids"]
        return n + sum(len(x) + 1 for x in enc)

    def _distribution(self, labels, scores):
        vals = [scores[label] for label in labels.values()]
        if self.probs == "softmax":  # undo the sigmoid, softmax over logits
            logits = [math.log(max(v, 1e-12)) - math.log(max(1 - v, 1e-12)) for v in vals]
            m = max(logits)
            vals = [math.exp(x - m) for x in logits]
        total = sum(vals)
        if not total or not math.isfinite(total):
            return {k: 1 / len(labels) for k in labels}
        return {k: v / total for k, v in zip(labels, vals)}

    def __call__(self, state, questions):
        body = render_state(state)
        keys, prompts, label_maps, lengths = [], [], [], []
        for key, q in questions.items():
            if q["type"] == "choice":
                labels = option_labels(q["criteria"])
            elif q["type"] == "noul":
                labels = option_labels(q.get("criteria") or NOUL_CRITERIA)
            else:
                raise Unsupported("Unsupported question type " + str(q["type"]))
            prompt = text(q.get("instructions", ""))
            n = self._n_tokens(body, prompt, labels.values())
            if n > self.max_length:
                raise Unsupported(f"input of {n} tokens exceeds max_length {self.max_length}")
            keys.append(key)
            prompts.append(prompt)
            label_maps.append(labels)
            lengths.append(n)

        # multi-label with threshold 0 returns the sigmoid score of every label
        # chunks keep (questions x longest input) under batch_tokens, so long states don't run out of memory
        results, start = [], 0
        while start < len(keys):
            end, longest = start, 0
            while end < len(keys) and end - start < self.batch_size:
                longest_next = max(longest, lengths[end])
                if end > start and longest_next * (end - start + 1) > self.batch_tokens:
                    break
                longest, end = longest_next, end + 1
            results += self.pipe([body] * (end - start), [list(m.values()) for m in label_maps[start:end]],
                                 threshold=0.0, batch_size=end - start, prompt=prompts[start:end],
                                 classification_type="multi-label")
            start = end
        answers, raw = {}, {}
        for key, labels, preds in zip(keys, label_maps, results):
            scores = {p["label"]: float(p["score"]) for p in preds}
            raw[key] = scores
            dist = self._distribution(labels, scores)
            if questions[key]["type"] == "noul":
                answers[key] = {"type": "noul", "noul": dist["true"]}
            else:
                answers[key] = {"type": "choice", "choice": max(dist, key=dist.get), "probabilities": dist}
        return {"model": self.model_id, "answers": answers}, raw
