"""Decision Index adapter for Wyat's native typed decision head.

Set PYTHONPATH to the model release's runtime/ directory, and pass the local
model/ directory with --model. Loading verifies the native artifact's hashes.
"""

import hashlib
import importlib.metadata
import json
from pathlib import Path

from decision_index.engines import Engine, Unsupported


class WyatEngine(Engine):
    name = "wyat-1-800m"
    latency = (
        "Single in-process native predict call, including input rendering, "
        "tokenization, every question and option, decision head and typed "
        "output conversion; device synchronized; excludes model loading."
    )

    def __init__(self, model, device="cuda", **options):
        if options:
            raise TypeError(
                "Unknown Wyat engine options: " + ", ".join(sorted(options))
            )
        super().__init__(model=model, device=device)
        import torch
        import wyat

        self.torch = torch
        self.wyat = wyat
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("Wyat's option-fork runtime supports cpu or cuda")
        manifest_path = Path(model) / "fork_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("input_rendering") is not None:
            raise ValueError(
                "Official evaluation requires uniform inputs; first run "
                "examples/prepare_wyat.py to disable the release's task-specific renderer"
            )
        self.model = wyat.load(
            model,
            backend="torch",
            device=device,
            attention="sdpa",
            compile=False,
            temperature=1.0,
        )
        self.provenance = {
            "model_repo": "teilomillet/wyat-1-800m",
            "model_id": self.model.model_id,
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "artifact": manifest,
            "runtime_files_sha256": {
                str(path.relative_to(Path(wyat.__file__).parent)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in sorted(Path(wyat.__file__).parent.rglob("*.py"))
            },
            "settings": {
                "temperature": 1.0,
                "attention": "sdpa",
                "compile": False,
                "request_batch_size": 1,
                "permutations": 1,
                "dtype": "native: CUDA BF16 backbone linears; FP32 adapters/head",
                "input_rendering": manifest.get("input_rendering"),
                "input_policy": "verbatim-public-record-v1",
            },
            "capacity_limits": {
                "choice_options": [2, 255],
                "noul_options": ["false", "true"],
                "max_branch_tokens": manifest["max_length"],
                "max_batch_tokens": manifest["max_batch_tokens"],
                "max_segment_utf8_bytes": manifest["max_length"] * 32,
                "max_branch_batch_size": 32,
                "truncation": False,
            },
        }

    def __call__(self, state, questions):
        for q in questions.values():
            if q["type"] not in {"choice", "noul"}:
                raise Unsupported("Decision Index adapter supports choice and noul")
            if q["type"] == "choice" and not 2 <= len(q["criteria"]) <= 255:
                raise Unsupported("Wyat requires 2..255 complete choice options")
        try:
            native = self.model.predict({"state": state, "questions": questions})
        except self.wyat.ModelInputError as error:
            if error.reason in {"input-length", "candidate-count", "unsupported-task"}:
                raise Unsupported("Wyat declared capacity: " + error.reason) from error
            raise
        answers = {}
        for key, q in questions.items():
            answer = native[key]
            if q["type"] == "choice":
                answers[key] = {
                    "type": "choice",
                    "choice": answer.choice,
                    "probabilities": answer.probabilities_by_id,
                }
            else:
                answers[key] = {"type": "noul", "noul": answer.probability_true}
        return {"model": self.model.model_id, "answers": answers}, None

    def synchronize(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def runtime(self):
        return {
            "python_packages": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "tokenizers", "safetensors")
            },
            "cuda": self.torch.version.cuda,
            "gpu": (
                self.torch.cuda.get_device_name(self.device)
                if self.device.type == "cuda"
                else None
            ),
        }

    def close(self):
        del self.model
