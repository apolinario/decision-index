"""Decision Index engine for DecisionTune 1.0 (decision-tune/decisiontune-1.0).

A thin kit adapter around engine.py from the model repository, which holds all the inference code.
One encoder pass per question: [CLS] question [SEP] ([MASK] option)+ [SEP] state [SEP]. Probabilities are the softmax
of the head score at each [MASK]; a noul question is scored as the options "yes" and "no" and answered with P(yes).
No calibration temperature, no option filtering, one fixed rendering. A sequence over the 8,192-token context is
refused as Unsupported, never truncated.

    python -m decision_index pipeline --engine decisiontune_engine:DecisionTuneEngine --edition 0.2.1 --out runs/decisiontune-1.0
    (run with PYTHONPATH=submissions/decisiontune-1.0; options: --option backend=auto|torch|mlx|onnx, --option path=<local model dir>)
"""
import os
import sys

from decision_index.engines.base import Engine, Unsupported

REPO = "decision-tune/decisiontune-1.0"
REVISION = "27056cb08bcf2b3ba1fd9bccb782b4dfd0498e1f"


class DecisionTuneEngine(Engine):
    name = "decisiontune-1.0"
    latency = "In-process request wall time incl. tokenization and prompt construction; excludes model loading. One encoder pass per question."

    def __init__(self, path=None, revision=REVISION, backend="auto", device=None, dtype="float32", **options):
        super().__init__(**options)
        if path:  # a local copy of the model files; revision is then not checked
            revision = None
        else:
            from huggingface_hub import snapshot_download
            path = snapshot_download(REPO, revision=revision)
        self.path = os.path.expanduser(path)
        sys.path.insert(0, self.path)
        from engine import DecisionModel, option_text  # engine.py from the model repository

        self.option_text = option_text
        self.m = DecisionModel(self.path, backend=backend, device=device, dtype=dtype)
        self.provenance = {"kind": "encoder", "repo": REPO, "revision": revision, "path": self.path,
                           "backend": self.m.backend_name, "device": self.m.device, "dtype": dtype, "context_limit_tokens": self.m.limit,
                           "policy": "Softmax over the head score at each option marker; no temperature, no truncation, no option filtering, one fixed rendering."}

    def runtime(self):
        return {"backend": self.m.backend_name, "device": self.m.device}

    def __call__(self, state, questions):
        built = {}
        for k, q in questions.items():  # any Unsupported refuses the whole request
            if q["type"] == "choice":
                if len(q["criteria"]) < 2:
                    raise Unsupported("choice question with fewer than 2 options")
                opts = [self.option_text(o, d) for o, d in q["criteria"].items()]
            elif q["type"] == "noul":
                opts = ["yes", "no"]
            else:
                raise Unsupported(f"unsupported question type {q['type']}")
            try:
                built[k] = self.m.encode(state, q.get("instructions", ""), opts)
            except ValueError as e:  # the context limit
                raise Unsupported(str(e))
        answers, raw = {}, {}
        for k, (ids, pos) in built.items():
            z = self.m.backend.logits(ids, pos)
            p = self.m._softmax(z)
            raw[k] = z
            q = questions[k]
            if q["type"] == "choice":
                probs = dict(zip(q["criteria"], p))
                answers[k] = {"type": "choice", "choice": max(probs, key=probs.get), "probabilities": probs}
            else:
                answers[k] = {"type": "noul", "noul": p[0]}
        return {"model": self.name, "answers": answers, "usage": {"input_tokens": sum(len(b[0]) for b in built.values())}}, raw
