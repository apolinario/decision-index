import gzip
import json

from decision_index.runner import run
from decision_index.scoring.report import load_results


def write_rows(path, n):
    with gzip.open(path, "wt") as f:
        for i in range(n):
            row = {"id": f"x:{i}", "family": "x", "split": "t", "state": {"i": i}, "questions": {"q1": {"type": "choice", "instructions": "pick", "criteria": {"A": "a", "B": "b"}}}, "expected": {"q1": "A"}, "_evaluation": {"run_id": f"24:x:{i}", "catalog_id": 24, "dataset": "MMLU", "group_id": f"x:{i}", "track": "x", "source_path": "x", "payload_sha256": "0", "proxy_tokens": 1, "benchmark_origin": "test"}}
            f.write(json.dumps(row) + "\n")


def test_run_resumes_and_keeps_row_format(tmp_path):
    rows = tmp_path / "rows.jsonl.gz"
    write_rows(rows, 5)
    out = tmp_path / "run"
    first = run("random", {"seed": 3}, rows, out, limit=2, log=lambda s: None)
    assert first["completed"] == 2
    second = run("random", {"seed": 3}, rows, out, log=lambda s: None)
    assert second["completed"] == 5
    results = load_results(out / "results.jsonl")
    assert len(results) == 5
    raw = [json.loads(l) for l in (out / "results.jsonl").open()]
    keys = {"run_id", "catalog_id", "dataset", "group_id", "track", "source_path", "payload_sha256", "proxy_tokens", "benchmark_origin", "payload", "started_utc", "engine", "status", "response", "raw_output", "completed_utc", "total_wall_ms", "model_request_wall_ms"}
    assert keys <= set(raw[0])
    assert (out / "status.json").exists() and (out / "environment.json").exists()
