"""Decision Index engine for wyat-2-1.4b, a ``wyat-ouro-decision-v2`` artifact.

Wyat's public loader runs the artifact with the context execution and numerics
its manifest declares (``execution.context_prefill``, ``execution.numerics``).
Every request goes unchanged to ``model.predict``, preserving question IDs and
option order. A Choice answer carries every option's probability and the
selected key; a Noul answer carries the probability of true.
"""

import hashlib
import importlib.metadata
from pathlib import Path

from decision_index.engines import Engine, Unsupported

QUESTION_TYPES = ("choice", "noul")
CHOICE_OPTIONS = (2, 255)
CAPACITY_REASONS = ("input-length", "candidate-count", "unsupported-task")


class Wyat2Engine(Engine):
    name = "wyat-2-1.4b"
    latency = (
        "Single in-process wyat predict call: input encoding, the artifact's "
        "declared context prefill, every candidate, the scalar head and typed "
        "output conversion; device synchronized; excludes model loading."
    )

    def __init__(self, model, device="cuda", **options):
        if options:
            raise TypeError(
                "Unknown wyat-2 engine options: " + ", ".join(sorted(options))
            )
        super().__init__(model=model, device=device)
        import torch
        import wyat
        from wyat.loading import ouro_artifact

        self.torch = torch
        self.wyat = wyat
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("wyat's Ouro runtime here supports cpu or cuda")
        manifest_path = Path(model) / ouro_artifact.MANIFEST
        manifest = ouro_artifact.validate_artifact(model)
        if manifest["format"] != ouro_artifact.FORMAT_V2:
            raise ValueError(
                "wyat-2 needs a wyat-ouro-decision-v2 artifact, which declares "
                "the context execution it was trained with"
            )
        if manifest["prompt"]["input_rendering"] is not None:
            raise ValueError(
                "Official evaluation requires uniform inputs; this artifact "
                "declares a task-specific input renderer"
            )
        self.model = wyat.load(
            model, backend="torch", device=device, attention="sdpa", temperature=1.0
        )
        execution = manifest["execution"]
        runtime_root = Path(wyat.__file__).parent
        self.provenance = {
            "model_repo": "teilomillet/wyat-2-1.4b",
            "model_id": self.model.model_id,
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "artifact": {
                key: manifest[key]
                for key in ("format", "model_id", "contract", "base", "execution")
            },
            "runtime_files_sha256": {
                str(path.relative_to(runtime_root)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in sorted(runtime_root.rglob("*.py"))
            },
            "settings": {
                "temperature": 1.0,
                "attention": execution["attention"],
                "context_prefill": execution["context_prefill"],
                "numerics": execution["numerics"],
                "candidate_batching": "shared_context",
                "prefix_cache": 0,
                "request_batch_size": 1,
                "permutations": 1,
                "dtype": "CUDA BF16 backbone under autocast; FP32 adapters and head",
                "input_rendering": None,
                "input_policy": "verbatim-public-record-v1",
            },
            "capacity_limits": {
                "question_types": list(QUESTION_TYPES),
                "choice_options": list(CHOICE_OPTIONS),
                "noul_options": ["false", "true"],
                "max_sequence_tokens": execution["max_length"],
                "truncation": False,
            },
        }

    def __call__(self, state, questions):
        for question in questions.values():
            if question["type"] not in QUESTION_TYPES:
                raise Unsupported("Decision Index adapter supports choice and noul")
            low, high = CHOICE_OPTIONS
            if (
                question["type"] == "choice"
                and not low <= len(question["criteria"]) <= high
            ):
                raise Unsupported("wyat-2 requires 2..255 complete choice options")
        try:
            native = self.model.predict({"state": state, "questions": questions})
        except self.wyat.ModelInputError as error:
            if error.reason in CAPACITY_REASONS:
                raise Unsupported(
                    "wyat-2 declared capacity: " + error.reason
                ) from error
            raise
        answers = {}
        for key, question in questions.items():
            answer = native[key]
            if question["type"] == "choice":
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
        packages = {}
        for name in ("torch", "transformers", "tokenizers", "safetensors"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        return {
            "python_packages": packages,
            "torch": self.torch.__version__,
            "cuda": self.torch.version.cuda,
            "gpu": (
                self.torch.cuda.get_device_name(self.device)
                if self.device.type == "cuda"
                else None
            ),
        }

    def close(self):
        del self.model
