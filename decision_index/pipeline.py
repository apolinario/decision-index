import json
import os
from datetime import datetime, timezone
from pathlib import Path

from decision_index import constants as C
from decision_index.scoring.index import index_entry, rnd, score_panel
from decision_index.scoring.report import benchmark_summary, load_results
from decision_index.suite.io import Suite, atomic_json


def score_run(suite, results_path, engine, out_dir, reference_results=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = load_results(results_path)
    reference = load_results(reference_results) if reference_results else None
    summary = benchmark_summary(suite, results, engine, reference)
    atomic_json(out / "benchmark-summary.json", summary)
    scored = score_panel(suite, results)
    index = index_entry(scored)
    atomic_json(out / "index.json", index)
    benchmarks = {}
    for b in summary["benchmarks"]:
        n = b["catalog_id"]
        entry = {k: (rnd(v) if isinstance(v, float) else v) for k, v in b.items() if k not in ("detail",)}
        entry["median_ms"] = rnd(b.get("median_ms"), 1)
        if n in C.HEADLINE and n in scored:
            h = index["benchmarks"][str(n)]
            entry.update(score=h["raw"], metric=C.HEADLINE_METRIC, tracks=h["tracks"], answered=round(h["coverage"] * b["requests"]) if b.get("requests") else b.get("answered"))
        entry["in_index"] = str(n) in index["benchmarks"] and index["benchmarks"][str(n)]["in_index"]
        benchmarks[str(n)] = entry
    completed = sum(summary["counts"].values())
    scores = {
        "engine": engine,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "suite": {"requests": C.SUITE_REQUESTS, "scoreable": C.SUITE_SCOREABLE_REQUESTS, "excluded": C.SUITE_EXCLUDED_REQUESTS, "benchmarks": C.SUITE_BENCHMARKS, "rows_sha256": C.SUITE_ROWS_GZ_SHA256},
        "completed": completed,
        "complete": completed >= C.SUITE_SCOREABLE_REQUESTS,
        "counts": summary["counts"],
        "latency_ms": {k: rnd(v, 1) for k, v in summary["successful_request_latency_ms"].items()},
        "decision_index": index["index"],
        "scores": index["scores"],
        "areas": index["areas"],
        "index_benchmarks": index["benchmarks"],
        "benchmarks": benchmarks,
        "frozen_panel": index["frozen_panel"],
        "formulas": index["formulas"],
        "panel_id": index["panel_id"],
    }
    atomic_json(out / "scores.json", scores)
    return scores


def upload_run(repo_id, run_dir, path_in_repo="", private=True, token=None, compress=True):
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    import gzip
    import shutil

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    run_dir = Path(run_dir)
    staged = run_dir / "_upload"
    staged.mkdir(exist_ok=True)
    for name in ("benchmark-summary.json", "index.json", "scores.json", "environment.json", "status.json"):
        if (run_dir / name).exists():
            shutil.copyfile(run_dir / name, staged / name)
    results = run_dir / "results.jsonl"
    if results.exists():
        if compress:
            with results.open("rb") as src, gzip.open(staged / "results.jsonl.gz", "wb", compresslevel=6) as dst:
                shutil.copyfileobj(src, dst)
        else:
            shutil.copyfile(results, staged / "results.jsonl")
    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(staged), path_in_repo=path_in_repo or ".", commit_message=f"decision-index results {run_dir.name}")
    shutil.rmtree(staged)
    return f"https://huggingface.co/datasets/{repo_id}"
