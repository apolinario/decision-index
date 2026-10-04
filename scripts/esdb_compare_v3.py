"""Refresh analysis from retained predictions without rerunning inference."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decision_index.cyber.build import load_partition
from decision_index.cyber.common import rows, write_json, file_hash
from decision_index.cyber.v3.scoring import score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="benchmarks/esdb-v0.3")
    p.add_argument("--baselines", default="runs/esdb-v03-baselines")
    p.add_argument("--sev-results")
    p.add_argument("--jev-results", help="Pinned Jev results from scripts/esdb_jev.py")
    p.add_argument("--out", default="runs/esdb-v03-comparison.json")
    a = p.parse_args()
    result = {"dataset_manifest_sha256": file_hash(Path(a.dataset) / "manifest.json"), "test_scored": False,
              "review_status": "independent security reviewers pending", "splits": {}}
    sources = {name: path for name, path in (("Sev-4B-raw", a.sev_results), ("Jev-1.13.0", a.jev_results)) if path}
    predictions_by_engine = {name: list(rows(path)) for name, path in sources.items()}
    result["prediction_sources"] = {name: {"path": path, "sha256": file_hash(path)} for name, path in sources.items()}
    for split in ("development", "calibration"):
        values = load_partition(a.dataset, split)
        runs = {r["_evaluation"]["run_id"] for r in values}
        reports = {}
        for path in sorted((Path(a.baselines) / split).glob("*.jsonl")):
            predictions = list(rows(path))
            selected = {r["run_id"] for r in predictions}
            reports[path.stem] = score([r for r in values if r["_evaluation"]["run_id"] in selected], predictions)
        for name, predictions in predictions_by_engine.items():
            reports[name] = score(values, [r for r in predictions if r["run_id"] in runs])
        result["splits"][split] = reports
    write_json(a.out, result)
    print({s: {name: {t: {k: r[k] for k in ("cases", "case_exact_accuracy", "case_group_macro_accuracy", "case_coverage")}
        for t, r in report["ranking_tracks"].items()} for name, report in reports.items()} for s, reports in result["splits"].items()})


if __name__ == "__main__":
    main()
