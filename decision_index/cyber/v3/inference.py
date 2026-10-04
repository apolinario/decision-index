"""Answer-free native Sev pointer inference; complete failures count as failures."""
import json
import time
from pathlib import Path

from ..common import digest, file_hash, rows, write_json
from .common import REVISION, payload_hash

CONTEXT = {"max_state": 8192, "max_branch": 8192, "max_packed": 16384}


def preflight(input_path):
    from ..gate import require
    for row in rows(input_path):
        require([row])
        if "expected" in row or "metadata" in row or "provenance" in row:
            raise ValueError("Model input contains label-sidecar fields")
        if row["split"] not in ("development", "calibration", "test"):
            raise ValueError("Unknown inference partition")
        payload = {"state": row["state"], "questions": row["questions"]}
        if payload_hash(**payload) != row["_evaluation"]["payload_sha256"]:
            raise ValueError("Input receipt changed")


def evaluate(input_path, out, *, device="cuda", backend="torch", checkpoint=None, log=print):
    preflight(input_path)
    return _evaluate(input_path, out, device=device, backend=backend, checkpoint=checkpoint, log=log)


def _evaluate(input_path, out, *, device="cuda", backend="torch", checkpoint=None, log=print):
    requested = f"macmacmacmac/Sev-4B@{REVISION}"
    if checkpoint is not None and checkpoint != requested:
        raise ValueError("This evaluation protocol pins the complete Sev checkpoint revision")
    import torch
    from kev.api import SystemOneRequest, to_record, question_keys
    from kev.checkpoint import Checkpoint, LoadOptions
    from kev.device import sync
    from kev.model import ContextOverflow
    out = Path(out)
    if out.exists():
        raise FileExistsError("Model run output already exists")
    out.mkdir(parents=True)
    ck = Checkpoint(requested)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    options = LoadOptions(dtype=torch.float32, merge=True, temperature=1.0, backend=backend)
    tokenizer, model = ck.load(device, options)
    import importlib.metadata
    provenance = {"libraries": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "safetensors")},
        "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name() if device == "cuda" else device,
        "inference_code_sha256": file_hash(__file__),"checkpoint": ck.requested, "revision": REVISION, "base": ck.meta.base, "base_revision": ck.meta.base_revision,
        "adapter_sha256": file_hash(ck.file("adapter_model.safetensors")), "head_sha256": file_hash(ck.file("head.pt")),
        "backend": backend, "device": device, "dtype": "fp32" if backend == "torch" else "stored bf16 MLX",
        "merge": True, "temperature": 1.0, "rotations": 1, "date_facts": False, "context": CONTEXT,
        "rendering": "kev.api.SystemOneRequest/to_record; pointer head; no generation", "input_sha256": file_hash(input_path),
        "api_sha256": file_hash(Path(__import__("kev.api", fromlist=["x"]).__file__)),
        "model_code_sha256": file_hash(Path(__import__("kev.model", fromlist=["x"]).__file__)),
        "checkpoint_code_sha256": file_hash(Path(__import__("kev.checkpoint", fromlist=["x"]).__file__)),
        "training_performed": False, "truncation": False}
    write_json(out / "provenance.json", provenance)
    count, failures, total_ms = 0, 0, 0
    with (out / "results.jsonl").open("x") as f:
        for row in rows(input_path):
            # The inference process receives no answer sidecars. to_record's
            # dummy label is an API-format field and is never consumed by forward.
            if "expected" in row or "metadata" in row or "provenance" in row:
                raise ValueError("Model input contains label-sidecar fields")
            payload = {"state": row["state"], "questions": row["questions"]}
            if payload_hash(**payload) != row["_evaluation"]["payload_sha256"]:
                raise ValueError("Input receipt changed")
            result = {**row["_evaluation"], "id": row["id"], "split": row["split"], "engine": "Sev-4B"}
            try:
                record, meta = to_record(SystemOneRequest.model_validate(payload))
                enc = model.encode(tokenizer, record, max_state=CONTEXT["max_state"], max_branch=CONTEXT["max_branch"], strict=True)
                if len(enc["ids"]) > CONTEXT["max_packed"]:
                    raise ContextOverflow("Complete request exceeds frozen packed context limit")
                sync(device)
                start = time.perf_counter()
                with torch.no_grad():
                    logits = model.forward(enc)
                sync(device)
                elapsed = (time.perf_counter() - start) * 1000
                total_ms += elapsed
                answers, raw = {}, {}
                for info, vector in zip(meta, logits):
                    z = vector.float().cpu()
                    keys = question_keys(row["questions"][info["id"]]["type"], row["questions"][info["id"]]["criteria"])
                    distribution = dict(zip(keys, torch.softmax(z, -1).tolist()))
                    answers[info["id"]] = {"type": "choice", "choice": max(distribution, key=distribution.get), "probabilities": distribution}
                    raw[info["id"]] = dict(zip(keys, z.tolist()))
                result.update(status="ok", response={"answers": answers}, logits=raw, latency_ms=elapsed, input_tokens=len(enc["ids"]))
            except ContextOverflow as error:
                failures += 1
                result.update(status="context_overflow", error=str(error))
            # Other runtime errors stop the run; they are not silently turned
            # into predictions, and the same completed test cannot be rerun.
            f.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
            count += 1
            if count % 50 == 0:
                log(f"Sev: {count} attempted, {failures} context failures, mean inference {total_ms / max(1, count-failures):.0f} ms", flush=True)
    write_json(out / "completion.json", {"attempted": count, "context_failures": failures, "results_sha256": file_hash(out / "results.jsonl"),
        "input_sha256": provenance["input_sha256"], "complete_attempts": True})
    return {"attempted": count, "context_failures": failures, "output": str(out)}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda", choices=("cuda", "mps", "cpu"))
    p.add_argument("--backend", default="torch", choices=("torch", "mlx"))
    args = p.parse_args()
    print(evaluate(args.input, args.out, device=args.device, backend=args.backend))
