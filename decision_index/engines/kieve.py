"""Native in-process engine for the Kieve-JeV encoder (scairesearch/kieve).

Kieve is a typed decision encoder: a state/question/pair encoder with a choice
head over the supplied options and a separate no-answer ("noul") head. This
adapter maps the Decision Index question schema (``{key: {"type", "instructions",
"criteria"}}``) onto Kieve's ``Case``/``Question`` API and runs the model in
process, one forward pass per request.

Requires the Kieve repo importable on ``PYTHONPATH`` (or installed). Load with::

    --engine decision_index.engines.kieve:KieveEngine \
    --option checkpoint=/path/to/checkpoint_latest.pt \
    --option config=/path/to/gpu-main-jevs.json \
    --option device=cuda

``--option model=<hub-repo>`` may be used instead of ``checkpoint``/``config`` to
pull ``checkpoint_latest.pt`` and ``config.json`` from the Hub.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

from decision_index.engines.base import Engine, Unsupported, text


class KieveEngine(Engine):
    name = "kieve"
    latency = "Device-synchronized in-process request wall time including prompt construction and a single batched forward pass; excludes model loading."

    def __init__(self, checkpoint=None, config=None, model=None, revision=None,
                 device=None, temperature=None, variant=None, max_state=None,
                 max_question=None, max_pair=None, max_options=None, **options):
        super().__init__(**options)
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        import torch

        from kieve.data import Case, Collate, HashTokenizer, HFTokenizer, Question
        from kieve.eval import run_model
        from kieve.model import build_model, model_config, to_device

        self.torch = torch
        self.Case = Case
        self.Question = Question
        self.run_model = run_model

        if model and not (checkpoint and config):
            from huggingface_hub import hf_hub_download

            checkpoint = checkpoint or hf_hub_download(model, "checkpoint_latest.pt", revision=revision)
            config = config or hf_hub_download(model, "config.json", revision=revision)
        if not checkpoint or not config:
            raise ValueError("KieveEngine needs --option checkpoint=<path> and --option config=<path>, or --option model=<hub-repo>")

        cfg = json.loads(Path(config).read_text())
        p, fixed = cfg.get("params", {}), cfg.get("fixed", {})
        self.model_id = model or str(Path(checkpoint).parent)
        self.device = torch.device(device or fixed.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.variant = str(variant or p.get("variant", "jev")).lower()

        tok_spec = str(p.get("tokenizer", "hash"))
        if tok_spec.startswith("hf:"):
            tokenizer = HFTokenizer(tok_spec[3:])
            vocab_size = len(tokenizer.tok)
        else:
            from bertbm.tokenizer import Vocab

            tokenizer = HashTokenizer(Vocab(base_vocab=int(p.get("vocab_size", 262))))
            vocab_size = int(p.get("vocab_size", 262))

        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
        mc = model_config(str(fixed.get("config_class", "tiny")))
        mc.vocab_size = vocab_size
        self.model = build_model(mc, vocab_size, variant=self.variant,
                                 use_ops=bool(fixed.get("use_ops", True)))
        self.model.load_state_dict(ckpt["model_state"], strict=False)
        to_device(self.model, self.device)
        self.model.eval()
        self.temperature = float(temperature if temperature is not None else ckpt.get("temperature", 1.0))

        self.collate = Collate(
            tokenizer,
            max_state=int(max_state if max_state is not None else fixed.get("max_state", 224)),
            max_question=int(max_question if max_question is not None else fixed.get("max_question", 32)),
            max_pair=int(max_pair if max_pair is not None else fixed.get("max_pair", 48)),
            max_options=int(max_options if max_options is not None else fixed.get("max_options", 8)),
            variant=self.variant,
        )
        self.provenance = {
            "kind": "kieve",
            "repo": model or "local",
            "revision": revision,
            "checkpoint": None if model else str(checkpoint),
            "config": None if model else str(config),
            "device": str(self.device),
            "variant": self.variant,
            "temperature": self.temperature,
            "policy": "State, question and one row per option are encoded in a single forward pass; the choice distribution is softmax over the option logits and the noul probability is sigmoid(noul_logit). All supplied options are kept; no option is filtered or truncated.",
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

    def _to_question(self, key, q):
        qtype = str(q.get("type", "choice")).lower()
        if qtype == "choice":
            criteria = q.get("criteria") or {}
            keys = list(criteria)
            if not keys:
                raise Unsupported("choice question without options")
            return keys, self.Question(
                qid=str(key), qtype="choice",
                text=str(q.get("instructions", "")),
                options=[text(criteria[k]) for k in keys],
            )
        if qtype == "noul":
            return ["false", "true"], self.Question(
                qid=str(key), qtype="noul",
                text=str(q.get("instructions", "")),
            )
        raise Unsupported("Unsupported question type " + str(qtype))

    def __call__(self, state, questions):
        torch = self.torch
        parsed, qs = [], []
        for key, q in questions.items():
            keys, kq = self._to_question(key, q)
            parsed.append((key, q, keys))
            qs.append(kq)

        rows = [(self.Case("req", text(state) if state not in ("", None) else "", qs, "request"), q) for q in qs]
        batch = self.collate(rows).to(self.device)
        with torch.no_grad():
            out = self.run_model(self.model, batch, self.variant)
        logits = (out["logits"] / self.temperature).detach().cpu().numpy()
        noul_p = torch.sigmoid(out["noul_logit"] / self.temperature).detach().cpu().numpy()

        answers, raw = {}, {}
        for i, (key, q, keys) in enumerate(parsed):
            if q["type"] == "noul":
                value = float(min(max(noul_p[i], 0.0), 1.0))
                if not math.isfinite(value):
                    value = 0.0
                answers[key] = {"type": "noul", "noul": value}
                raw[key] = {"noul_logit": float(out["noul_logit"][i].detach().cpu())}
                continue
            row = logits[i]
            n = min(len(keys), row.shape[0])
            row = row[:n]
            row = [v if math.isfinite(v) else -1e9 for v in row.tolist()]
            top = max(row)
            exps = [math.exp(v - top) for v in row]
            z = sum(exps)
            probs = {keys[j]: exps[j] / z for j in range(n)}
            choice = max(probs, key=lambda x: probs[x])
            answers[key] = {"type": "choice", "choice": choice, "probabilities": probs}
            raw[key] = {"option_logits": dict(zip(keys, row))}

        return {"model": self.model_id, "answers": answers}, raw
