import collections
import hashlib
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from decision_index import constants as C
from decision_index.engines import NativeAbstention, Unsupported, load_engine, validate
from decision_index.suite.io import atomic_json, dumps, read_jsonl, sha256_file


def stamp():
    return datetime.now(timezone.utc).isoformat()


def iter_rows(rows_path, keep=None):
    paths = rows_path if isinstance(rows_path, (list, tuple)) else [rows_path]
    for path in paths:
        for row in read_jsonl(path):
            if keep is None or keep(row["_evaluation"]):
                yield row


def run(engine_name, engine_options, rows_path, out_dir, limit=None, compact=False, resume=True, warm=True, seed=C.RUN_SEED, corpus_sha256=None, halt_on_device_error=True, log=print, keep=None):
    out = Path(out_dir)
    results_path = out / "results.jsonl"
    environment_path = out / "environment.json"
    if not resume and (results_path.exists() or environment_path.exists()):
        raise FileExistsError("Fresh runs require a new output directory")
    paths = rows_path if isinstance(rows_path, (list, tuple)) else [rows_path]
    selected = hashlib.sha256()
    for row in iter_rows(rows_path, keep):
        from decision_index.cyber.gate import require
        require([row])
        selected.update(dumps([row["_evaluation"]["run_id"], row["state"], row["questions"]]).encode())
        selected.update(b"\n")
    identity = {"schema": 1, "engine": engine_name, "engine_options": engine_options,
                "seed": seed, "frozen_corpus_sha256": corpus_sha256,
                "inputs": [{"path": str(Path(p).resolve()), "sha256": sha256_file(p)} for p in paths],
                "selected_requests_sha256": selected.hexdigest(),
                "runner_sha256": sha256_file(__file__),
                "engine_code_sha256": {p.name: sha256_file(p) for p in sorted((Path(__file__).parent / "engines").glob("*.py"))}}
    previous_environment = json.loads(environment_path.read_text()) if environment_path.exists() else None
    if results_path.exists() and not previous_environment:
        raise ValueError("Existing predictions lack a run identity; use a new output directory")
    if previous_environment and {k: previous_environment.get("run_identity", {}).get(k) for k in identity} != identity:
        raise ValueError("Resume model settings, input bytes, selection or evaluator changed; use a new output directory")
    out.mkdir(parents=True, exist_ok=True)

    def event(**kw):
        record = {"time": stamp(), **kw}
        log(dumps(record))
        atomic_json(out / "status.json", record, indent=None)

    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
    t = time.perf_counter()
    engine = load_engine(engine_name, **engine_options)
    engine.synchronize()
    identity["model_source"] = engine.provenance
    runtime = engine.runtime()
    identity["runtime"] = runtime
    if previous_environment and previous_environment["run_identity"] != identity:
        engine.close()
        raise ValueError("Resolved model artifacts or runtime changed; use a new output directory")
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if not previous_environment:
        atomic_json(environment_path, {"engine": engine_name, "engine_options": engine_options, "model_source": engine.provenance, **runtime, "loaded_seconds": time.perf_counter() - t, "frozen_corpus_sha256": corpus_sha256, "rows_path": [str(p) for p in paths], "runner_sha256": identity["runner_sha256"], "run_identity": identity, "run_identity_sha256": identity_hash, "latency": engine.latency})
    if warm:
        engine.warmup()
        engine.synchronize()
    event(event="ready", engine=engine_name)
    previous = {}
    if resume and results_path.exists():
        for r in read_jsonl(results_path, complete_lines_only=True):
            if r.get("run_identity_sha256") != identity_hash:
                engine.close()
                raise ValueError("Stored prediction belongs to a different run")
            previous[r["run_id"]] = r["status"]
        if previous:
            text = results_path.read_text(encoding="utf-8")
            if not text.endswith("\n"):
                results_path.write_text(text[: text.rfind("\n") + 1], encoding="utf-8")
    counts = collections.Counter()
    start = time.perf_counter()
    finished = 0
    completed = set(previous)
    with results_path.open("a", encoding="utf-8") as logf:
        for row in iter_rows(rows_path, keep):
            e = row["_evaluation"]
            rid = e["run_id"]
            if rid in previous and previous[rid] != "error":
                continue
            payload = {"state": row["state"], "questions": row["questions"]}
            t = time.perf_counter()
            engine.synchronize()
            result = {**e, "started_utc": stamp(), "engine": engine_name,
                      "run_identity_sha256": identity_hash,
                      "request_sha256": hashlib.sha256(dumps(payload).encode()).hexdigest()}
            if not compact:
                result["payload"] = payload
            try:
                response, raw = engine(**payload)
                engine.synchronize()
                validate(payload["questions"], response)
                result.update(status="ok", response=response)
                if not compact:
                    result["raw_output"] = raw
            except NativeAbstention as exc:
                result.update(status="abstained", error=str(exc))
                if not compact:
                    result["raw_output"] = exc.raw
            except Unsupported as exc:
                result.update(status="unsupported", error=str(exc))
            except Exception as exc:
                result.update(status="error", error=str(exc), exception=type(exc).__name__, traceback=traceback.format_exc())
            elapsed = (time.perf_counter() - t) * 1000
            result.update(completed_utc=stamp(), total_wall_ms=elapsed, model_request_wall_ms=elapsed)
            logf.write(dumps(result) + "\n")
            logf.flush()
            counts[result["status"]] += 1
            finished += 1
            completed.add(rid)
            if finished % 10 == 0:
                event(event="progress", engine=engine_name, completed=len(completed), counts=dict(counts), elapsed_seconds=round(time.perf_counter() - start, 1))
            if halt_on_device_error and (result.get("exception") in ("OutOfMemoryError", "AcceleratorError") or "device-side assert" in result.get("error", "")):
                event(event="failed", engine=engine_name, completed=len(completed), counts=dict(counts), error=result.get("error"), failed_run_id=rid, elapsed_seconds=round(time.perf_counter() - start, 1))
                raise RuntimeError("Device error; stopping before the accelerator context is reused")
            if counts["error"] >= 5 and counts["ok"] == 0:
                event(event="failed", engine=engine_name, completed=len(completed), counts=dict(counts), error="five failed requests without a success")
                raise RuntimeError("Five failed requests without a success; stop instead of filling the benchmark with runtime errors")
            if limit and finished >= limit:
                break
    engine.close()
    final = dict(event="complete", engine=engine_name, completed=len(completed), counts=dict(counts), elapsed_seconds=round(time.perf_counter() - start, 1))
    event(**final)
    return final
