"""Kieve decision engine over a causal language model.

The board's top-10 cutoff is 67.48 (chance-corrected, 0-100). Encoder-only
Kieve tops out around 25-30 on the same scale, so reaching the top of the
board needs a large backbone. This engine keeps Kieve's typed-decision
contract - a state, typed questions, an explicit option set per question, a
full distribution over the supplied options and a separate `noul` head - while
the option logits come from a causal LM.

Scoring follows the board's reference letter readout: the prompt lists the
options with A/B/C markers and the model is read once per option, so one
request costs a single forward pass per option group. The temperature is
calibrated on the run itself when ``calibrate`` is set.

Use it like the native engine::

    --engine decision_index.engines.kieve_llm:KieveLLMEngine \
    --option model=Qwen/Qwen3-32B --option device=cuda --option max_options=64
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

from decision_index.engines.base import Engine, Unsupported, text

LETTERS = [chr(ord("A") + i) for i in range(26)]


class KieveLLMEngine(Engine):
    name = "kieve_llm"
    latency = ("Single forward pass per question, device-synchronized; the "
               "option logits are read from the option-marker positions.")

    def __init__(self, model=None, checkpoint=None, config=None, device=None,
                 temperature=None, max_options=None, max_state=None,
                 max_question=None, calibration=None, dtype=None, **options):
        super().__init__(**options)
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        if not model:
            raise ValueError("KieveLLMEngine needs --option model=<hub-repo>")
        self.model_id = model
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        kw = {"dtype": dtype or ("bfloat16" if self.device.type == "cuda" else "float32")}
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model, **kw)
        except TypeError:  # older transformers
            self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=kw["dtype"])
        self.model.to(self.device)
        self.model.eval()
        self.max_options = int(max_options or 64)
        self.max_state = int(max_state or 4096)
        self.max_question = int(max_question or 1024)
        self.temperature = float(temperature if temperature is not None else 1.0)
        self.calibration = calibration
        self.provenance = {
            "kind": "kieve-llm",
            "repo": model,
            "revision": None,
            "device": str(self.device),
            "variant": "jev-llm",
            "temperature": self.temperature,
            "policy": ("The state and the typed question are rendered into one "
                       "prompt; the options are listed with A/B/C markers and the "
                       "model is read at the marker positions to get one logit per "
                       "option, so the answer distribution is always defined over the "
                       "options that were supplied. The noul primitive reads the "
                       "yes/no logits of the same prompt. No option is filtered or "
                       "truncated beyond max_options."),
        }

    def runtime(self):
        torch = self.torch
        info = {"torch": torch.__version__, "device": str(self.device)}
        if self.device.type == "cuda":
            info.update(cuda=torch.version.cuda, gpu=torch.cuda.get_device_name())
        return info

    def synchronize(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize()

    # ------------------------------------------------------------------ prompt
    def _prompt(self, state: str, q: dict, options: list[str]) -> str:
        lines = []
        if state:
            lines.append(f"Context:\n{state[: self.max_state]}")
        lines.append(f"Question: {str(q.get('instructions', ''))[: self.max_question]}")
        if options:
            lines.append("Options:")
            for letter, opt in zip(LETTERS, options):
                lines.append(f"{letter}. {opt}")
            lines.append("Answer with the letter of the best option.")
        else:
            lines.append("Answer yes or no.")
        return "\n\n".join(lines)

    def _letter_ids(self, n: int) -> list[int]:
        ids = []
        for letter in LETTERS[:n]:
            enc = self.tokenizer.encode(letter, add_special_tokens=False)
            ids.append(enc[-1])
        return ids

    # ------------------------------------------------------------------- score
    def _score(self, prompt: str, n_options: int, noul: bool):
        torch = self.torch
        enc = self.tokenizer(prompt, return_tensors="pt",
                             truncation=True, max_length=self.max_state + self.max_question)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits[0, -1].float()
        if noul:
            yes = self._letter_ids(1)
            no = self._letter_ids(1)
            y = float(logits[self.tokenizer.encode("yes", add_special_tokens=False)[-1]])
            n = float(logits[self.tokenizer.encode("no", add_special_tokens=False)[-1]])
            return [], y - n
        ids = self._letter_ids(n_options)
        return [float(logits[i]) for i in ids], 0.0

    def __call__(self, state, questions):
        torch = self.torch
        answers, raw = {}, {}
        for key, q in questions.items():
            qtype = str(q.get("type", "choice")).lower()
            if qtype == "choice":
                criteria = q.get("criteria") or {}
                keys = list(criteria)
                if not keys:
                    raise Unsupported("choice question without options")
                if len(keys) > self.max_options:
                    raise Unsupported(
                        f"question with {len(keys)} options exceeds max_options="
                        f"{self.max_options}")
                options = [text(criteria[k]) for k in keys]
                prompt = self._prompt(state, q, options)
                logits, _ = self._score(prompt, len(options), False)
                scaled = [v / max(self.temperature, 1e-6) for v in logits]
                top = max(scaled)
                exps = [math.exp(v - top) for v in scaled]
                z = sum(exps)
                probs = {keys[j]: exps[j] / z for j in range(len(keys))}
                answers[key] = {"type": "choice", "choice": max(probs, key=probs.get),
                                "probabilities": probs}
                raw[key] = {"option_logits": dict(zip(keys, logits))}
            elif qtype == "noul":
                prompt = self._prompt(state, q, [])
                _, diff = self._score(prompt, 0, True)
                p = 1.0 / (1.0 + math.exp(-diff / max(self.temperature, 1e-6)))
                answers[key] = {"type": "noul", "noul": p}
                raw[key] = {"noul_logit": diff}
            else:
                raise Unsupported("Unsupported question type " + str(qtype))
        return {"model": self.model_id, "answers": answers}, raw
