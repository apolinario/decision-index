import argparse
import json
from pathlib import Path

from ..build import load_partition
from ..common import rows, write_json, write_rows, file_hash
from .build import build, verify
from .scoring import score


def main():
    p = argparse.ArgumentParser(description="ESDB v0.3 build, review, calibration and single-test protocol")
    commands = p.add_subparsers(dest="command", required=True)
    b = commands.add_parser("build")
    b.add_argument("--parent", default="benchmarks/esdb-v0.1")
    b.add_argument("--collection", default="../jevalin-collect")
    b.add_argument("--sev-root", default="../Sev-security-20260925")
    b.add_argument("--typescript", default="tools/esdb/node_modules/typescript/lib/typescript.js")
    b.add_argument("--out", default="benchmarks/esdb-v0.3")
    b.add_argument("--work", default="work/esdb-v0.3-build")
    for command in ("verify", "baselines", "stage-devcal", "score", "review", "calibrate", "seal"):
        sub = commands.add_parser(command)
        sub.add_argument("--dataset", default="benchmarks/esdb-v0.3")
        if command == "baselines":
            sub.add_argument("--out", required=True)
        elif command == "stage-devcal":
            sub.add_argument("--out", default="work/esdb-sev-v03-devcal.jsonl.gz")
        elif command == "score":
            sub.add_argument("--split", choices=("development", "calibration", "test"), required=True)
            sub.add_argument("--results", required=True)
            sub.add_argument("--out", required=True)
        elif command == "review":
            sub.add_argument("--profiles", nargs=2, required=True)
            sub.add_argument("--annotations", nargs=2, required=True)
            sub.add_argument("--packet", required=True, help="Strict blinded review packet, with annotation-derived provenance omitted")
            sub.add_argument("--out", required=True)
        elif command == "calibrate":
            sub.add_argument("--results", required=True)
            sub.add_argument("--out", required=True)
        elif command == "seal":
            sub.add_argument("--certificate", required=True)
            sub.add_argument("--proposal", required=True)
            sub.add_argument("--model-run", required=True)
            sub.add_argument("--baselines", required=True)
            sub.add_argument("--out", required=True)
    t = commands.add_parser("stage-test")
    t.add_argument("--protocol", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--input", default="work/esdb-sev-test.jsonl.gz")
    f = commands.add_parser("finish-test")
    f.add_argument("--protocol", required=True)
    f.add_argument("--model-run", required=True)
    f.add_argument("--out", required=True)
    a = p.parse_args()
    if a.command == "build":
        result = build(a.parent, a.collection, a.sev_root, a.typescript, a.out, a.work)
    elif a.command in ("stage-test", "finish-test"):
        from .final_test import stage, finish
        result = stage(a.protocol, a.out, a.input) if a.command == "stage-test" else finish(a.protocol, a.model_run, a.out)
    else:
        verify(a.dataset)
        if a.command == "verify":
            result = verify(a.dataset)
        elif a.command == "baselines":
            from .baselines import run
            result = run(load_partition(a.dataset, "development"), load_partition(a.dataset, "calibration"), a.out)
        elif a.command == "stage-devcal":
            if Path(a.out).exists():
                raise FileExistsError("Staged inputs are immutable")
            values = [r for s in ("development", "calibration") for r in rows(Path(a.dataset) / f"inputs/{s}.jsonl.gz")]
            write_rows(a.out, values)
            write_json(Path(a.out).with_suffix(".receipt.json"), {"dataset_manifest_sha256": file_hash(Path(a.dataset) / "manifest.json"),
                       "input_sha256": file_hash(a.out), "cases": len(values), "splits": ["development", "calibration"], "gold_exposed": False})
            result = {"cases": len(values), "input": a.out}
        elif a.command == "score":
            values = load_partition(a.dataset, a.split)
            run_ids = {r["_evaluation"]["run_id"] for r in values}
            predictions = [r for r in rows(a.results) if r["run_id"] in run_ids]
            result = score(values, predictions)
            write_json(a.out, result)
            result = {"report": a.out, "complete": result["complete"], "ranking_tracks": result["ranking_tracks"]}
        else:
            from .protocol import certify, calibrate, seal
            if a.command == "review":
                result = certify(a.dataset, a.profiles, a.annotations, a.out, a.packet)
            elif a.command == "calibrate":
                results = [r for r in rows(a.results) if r.get("split") == "calibration"]
                result = calibrate(load_partition(a.dataset, "calibration"), results, a.out)
                result = {"proposal": a.out, "temperatures": result["temperatures"], "status": result["status"]}
            else:
                result = seal(a.dataset, a.certificate, a.proposal, a.model_run, a.baselines, a.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
