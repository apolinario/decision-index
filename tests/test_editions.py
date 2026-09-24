import gzip
import json

from decision_index import editions
from decision_index.runner import run
from decision_index.scoring.report import load_results
from decision_index.suite.build.release_v2 import cut
from decision_index.suite.io import Suite, detect_edition


def row(n, group, i, run_id=None):
    return {"id": f"{n}:{group}:{i}", "state": {}, "questions": {"q": {"type": "choice", "instructions": "", "criteria": {"A": "a", "B": "b"}}}, "expected": {"q": "A"}, "_evaluation": {"run_id": run_id or f"{n}:{group}:{i}", "catalog_id": n, "dataset": "x", "group_id": group, "track": "t"}}


def write(path, rows):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_v2_cut_keeps_subset_and_excluded_groups(tmp_path):
    kept = editions.retrieval_subsets()["subsets"]["ToolRet"]["group_ids"][0]
    rows = [row(24, "m", 0), row(2, kept, 0), row(2, kept, 1), row(2, "dropped", 0), row(2, "carried", 0), row(2, "carried", 1), row(36, "gone", 0)]
    src = tmp_path / "v1.jsonl"
    write(src, rows)
    out = tmp_path / "v2.jsonl"
    report = cut(src, out, excluded={"2:carried:1"})
    kept_ids = [json.loads(l)["_evaluation"]["run_id"] for l in out.open()]
    assert kept_ids == ["24:m:0", f"2:{kept}:0", f"2:{kept}:1", "2:carried:0", "2:carried:1"]
    assert report["requests"] == 5 and report["toolret_requests"] == 4 and report["bright_requests"] == 0


def test_suite_02_applies_acos_subset_and_reads_added_rows(tmp_path):
    keep = sorted(editions.acos_subset()["run_ids"])[0]
    write(tmp_path / editions.ROWS_FILE, [row(38, "r1", 0, keep), row(38, "r2", 0, "38:dropped"), row(24, "m", 0)])
    write(tmp_path / editions.ADDED_FILE, [row(57, "p", 0, "candidates-v3:57:1")])
    (tmp_path / editions.MANIFEST_FILE).write_text(json.dumps({"edition": "release-v2"}))
    assert detect_edition(tmp_path) == "0.2"
    suite = Suite(tmp_path)
    assert [r["_evaluation"]["run_id"] for r in suite.rows()] == [keep, "24:m:0", "candidates-v3:57:1"]
    out = tmp_path / "run"
    run("random", {"seed": 1}, suite.row_paths, out, keep=suite.in_edition, warm=False, log=lambda s: None)
    assert sorted(load_results(out / "results.jsonl")) == sorted([keep, "24:m:0", "candidates-v3:57:1"])
    assert not suite.verify(strict=False)["match"]


def test_suite_01_detected_without_added_rows(tmp_path):
    write(tmp_path / editions.ROWS_FILE, [row(38, "r2", 0, "38:any")])
    suite = Suite(tmp_path)
    assert suite.edition["id"] == "0.1"
    assert [r["_evaluation"]["run_id"] for r in suite.rows()] == ["38:any"]
