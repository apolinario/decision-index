"""Evaluate pinned hosted Jev on frozen ESDB development/calibration requests."""
import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from decision_index.cyber.common import file_hash, rows
from decision_index.engines.base import validate
from decision_index.engines.http import CAPACITY_MARKERS
from decision_index.suite.io import atomic_json, dumps

MODEL = "jev-1.13.0"
BASE_URL = "https://api.typesafe.ai"


def stamp():
    return datetime.now(timezone.utc).isoformat()


def credential(path):
    for name in ("JEV_API_KEY", "TYPESAFE_API_KEY", "DECISION_INDEX_API_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    for line in Path(path).read_text().splitlines():
        name, sep, value = line.partition("=")
        if sep and name.strip().removeprefix("export ") in ("JEV_API_KEY", "TYPESAFE_API_KEY"):
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if value:
                return value
    raise ValueError("No Jev credential available")


def payload(row):
    return {"state": row["state"], "questions": row["questions"]}


def request_hash(row):
    return hashlib.sha256(dumps(payload(row)).encode()).hexdigest()


def load_inputs(source, dataset):
    receipt = json.loads(source.with_suffix(".receipt.json").read_text())
    if receipt["input_sha256"] != file_hash(source) or receipt["dataset_manifest_sha256"] != file_hash(dataset / "manifest.json"):
        raise ValueError("Staged input or dataset receipt changed")
    values = list(rows(source))
    native = [r for split in ("development", "calibration") for r in rows(dataset / f"inputs/{split}.jsonl.gz")]
    if len(values) != receipt["cases"] or len(values) != len(native):
        raise ValueError("Staged request count differs from frozen development/calibration")
    if not receipt.get("gold_exposed") is False:
        raise ValueError("Staging receipt must declare answer-free input")
    for row, frozen in zip(values, native):
        if set(row) != {"id", "family", "split", "state", "questions", "_evaluation"}:
            raise ValueError("Only answer-free staged requests are admitted")
        if row["split"] not in ("development", "calibration") or row != frozen:
            raise ValueError("Only unchanged frozen development/calibration requests are admitted")
    if len({r["_evaluation"]["run_id"] for r in values}) != len(values):
        raise ValueError("Repeated input request")
    return values, receipt


async def evaluate(client, row, retries=2):
    start = time.perf_counter()
    result = {**row["_evaluation"], "split": row["split"], "family": row["family"],
              "engine": MODEL, "request_sha256": request_hash(row), "started_utc": stamp()}
    attempts = []
    fatal = False
    for attempt in range(retries + 1):
        try:
            response = await client.post("/v1/systemone", json={"model": MODEL, **payload(row)})
        except httpx.TransportError as error:
            attempts.append({"attempt": attempt + 1, "exception": type(error).__name__})
            if attempt < retries:
                await asyncio.sleep(2 ** attempt)
                continue
            result.update(status="error", error=type(error).__name__)
            break
        attempts.append({"attempt": attempt + 1, "http_status": response.status_code,
                         "request_id": response.headers.get("x-typesafe-request-id")})
        if response.status_code in (429, 529, 500, 502, 503, 504) and attempt < retries:
            try:
                delay = min(30, max(2 ** attempt, float(response.headers.get("retry-after", "0"))))
            except ValueError:
                delay = 2 ** attempt
            await asyncio.sleep(delay)
            continue
        if not response.is_success:
            capacity = response.status_code in (400, 413, 422) and any(marker in response.text.lower() for marker in CAPACITY_MARKERS)
            result.update(status="unsupported" if capacity else "error", error=f"HTTP {response.status_code}")
            fatal = response.status_code in (401, 403, 404) or response.status_code == 422 and not capacity
            break
        try:
            raw = response.json()
            result["raw_output"] = raw
            if raw.get("model") != MODEL:
                result.update(status="error", error="Served model differs from pinned model")
                fatal = True
                break
            result["response"] = {k: v for k, v in raw.items() if k != "evaluation_trace"}
            validate(row["questions"], result["response"])
            usage = raw.get("usage", {}).get("input_tokens")
            if not isinstance(usage, int) or isinstance(usage, bool) or usage < 0:
                raise ValueError("Missing or invalid input token usage")
            result.update(status="ok", input_tokens=usage)
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            result.update(status="invalid", error=str(error))
        break
    result.update(attempts=attempts, completed_utc=stamp(), total_wall_ms=1000 * (time.perf_counter() - start))
    return result, fatal


def resume_results(path, values, identity_hash):
    by_id = {r["_evaluation"]["run_id"]: r for r in values}
    completed = {}
    if path.exists():
        data = path.read_bytes()
        if data and not data.endswith(b"\n"):
            raise ValueError("Incomplete result line; inspect before resuming")
        for result in rows(path):
            rid = result["run_id"]
            if rid in completed or rid not in by_id:
                raise ValueError("Repeated or foreign prediction")
            row = by_id[rid]
            if result.get("run_identity_sha256") != identity_hash or result.get("request_sha256") != request_hash(row) or result.get("payload_sha256") != row["_evaluation"]["payload_sha256"]:
                raise ValueError("Stored prediction identity or input changed")
            if result["status"] == "ok":
                if result["response"].get("model") != MODEL:
                    raise ValueError("Stored model differs from pin")
                validate(row["questions"], result["response"])
            completed[rid] = result
    return completed


async def run(args):
    values, receipt = load_inputs(args.input, args.dataset)
    identity = {"schema": 1, "model": MODEL, "base_url": BASE_URL, "input_sha256": receipt["input_sha256"],
                "dataset_manifest_sha256": receipt["dataset_manifest_sha256"], "concurrency": args.concurrency,
                "retries": 2, "timeout_seconds": 120, "max_reported_input_tokens": args.max_tokens,
                "python": sys.version, "httpx": importlib.metadata.version("httpx"),
                "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in (
                    Path(__file__), ROOT / "decision_index/engines/base.py", ROOT / "decision_index/engines/http.py",
                    ROOT / "decision_index/suite/io.py", ROOT / "decision_index/cyber/common.py")},
                "policy": "Unchanged evidence and question order; no labels, truncation, rotations or confidence threshold. Transport retries only; failures retained once. Raw probabilities preserved, standard scorer used."}
    identity_hash = hashlib.sha256(dumps(identity).encode()).hexdigest()
    provenance_path = args.out / "provenance.json"
    if provenance_path.exists() and json.loads(provenance_path.read_text())["run_identity"] != identity:
        raise ValueError("Run settings, input or code changed; use a new output directory")
    results_path = args.out / "results.jsonl"
    if results_path.exists() and not provenance_path.exists():
        raise ValueError("Existing results lack provenance")
    completed = resume_results(results_path, values, identity_hash)
    key = credential(args.env_file)
    args.out.mkdir(parents=True, exist_ok=True)
    if not provenance_path.exists():
        atomic_json(provenance_path, {"started_utc": stamp(), "run_identity": identity,
                    "run_identity_sha256": identity_hash, "model_revision": "Hosted version ID; provider weight hashes are not exposed",
                    "price_per_million_input_tokens_usd": .042, "pricing_source": "https://docs.typesafe.ai/models",
                    "requests": len(values), "questions": sum(len(r["questions"]) for r in values), "test_inference": False})
    pending = [r for r in values if r["_evaluation"]["run_id"] not in completed]
    if args.limit:
        pending = pending[:args.limit]
    reported_tokens = sum(r.get("raw_output", {}).get("usage", {}).get("input_tokens", 0) for r in completed.values())
    queue = asyncio.Queue()
    for row in pending:
        queue.put_nowait(row)
    stopped = asyncio.Event()
    counts = Counter(r["status"] for r in completed.values())
    started = time.perf_counter()

    def status():
        result = {"completed": len(completed), "expected": len(values), "counts": dict(counts),
                  "reported_input_tokens": reported_tokens, "estimated_usage_cost_usd": reported_tokens / 1e6 * .042,
                  "elapsed_seconds_this_process": round(time.perf_counter() - started, 1), "test_inference": False}
        atomic_json(args.out / "status.json", result)
        print(dumps(result), flush=True)
        return result

    with results_path.open("a", encoding="utf-8") as output:
        async with httpx.AsyncClient(base_url=BASE_URL, headers={"Authorization": "Bearer " + key}, timeout=120,
                                     limits=httpx.Limits(max_connections=args.concurrency)) as client:
            async def worker():
                nonlocal reported_tokens
                while not stopped.is_set():
                    try:
                        row = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    # Reserve the documented maximum for all in-flight requests.
                    if reported_tokens + args.concurrency * 64000 > args.max_tokens:
                        stopped.set()
                        return
                    result, fatal = await evaluate(client, row)
                    result["run_identity_sha256"] = identity_hash
                    output.write(dumps(result) + "\n")
                    output.flush()
                    os.fsync(output.fileno())
                    completed[result["run_id"]] = result
                    counts[result["status"]] += 1
                    reported_tokens += result.get("raw_output", {}).get("usage", {}).get("input_tokens", 0)
                    if fatal or counts["error"] >= 5 and counts["ok"] == 0:
                        stopped.set()
                    if len(completed) % 100 == 0 or fatal:
                        status()
            await asyncio.gather(*(worker() for _ in range(args.concurrency)))
    summary = status()
    summary.update(all_attempted=len(completed) == len(values), results_sha256=file_hash(results_path),
                   run_identity_sha256=identity_hash, ended_utc=stamp())
    atomic_json(args.out / "completion.json", summary)
    if stopped.is_set():
        raise RuntimeError("Stopped on a fatal provider response or token budget; retained results require inspection")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "work/esdb-sev-v03-devcal.jsonl.gz")
    parser.add_argument("--dataset", type=Path, default=ROOT / "benchmarks/esdb-v0.3")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/jev-v03-devcal-20261004")
    parser.add_argument("--env-file", type=Path, default=ROOT.parent / "jevalin/.env")
    parser.add_argument("--concurrency", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--max-tokens", type=int, default=20_000_000)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(dumps(asyncio.run(run(args))), flush=True)


if __name__ == "__main__":
    main()
