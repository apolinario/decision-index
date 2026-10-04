import argparse
import json
from pathlib import Path

from .build import build, load_partition, verify
from .common import SPLITS, file_hash, rows, write_json
from .scoring import score


def main():
    parser = argparse.ArgumentParser(description="Build, verify or score the local four-track cybersecurity research dataset.")
    commands = parser.add_subparsers(dest="command", required=True)
    b = commands.add_parser("build")
    b.add_argument("--collection", type=Path, default=Path("../jevalin-collect"))
    b.add_argument("--sev-root", type=Path, default=Path("../Sev-security-20260925"))
    b.add_argument("--out", type=Path, default=Path("benchmarks/esdb-v0.1"))
    b.add_argument("--work", type=Path, default=Path("work/esdb-v0.1-build"))
    b.add_argument("--typescript", type=Path, help="TypeScript 5.9.3 module for a fresh static-only policy reparse.")
    v = commands.add_parser("verify")
    v.add_argument("--dataset", type=Path, default=Path("benchmarks/esdb-v0.1"))
    s = commands.add_parser("score")
    s.add_argument("--dataset", type=Path, default=Path("benchmarks/esdb-v0.1"))
    s.add_argument("--split", choices=SPLITS, default="test")
    s.add_argument("--results", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = build(args.collection, args.sev_root, args.out, args.work, typescript=args.typescript)
    elif args.command == "verify":
        result = verify(args.dataset)
    else:
        verify(args.dataset)
        result = score(load_partition(args.dataset, args.split), list(rows(args.results)))
        result.update(dataset_manifest_sha256=file_hash(args.dataset / "manifest.json"), split=args.split,
                      results_sha256=file_hash(args.results))
        write_json(args.out, result)
        result = {"report": str(args.out), "complete": result["complete"],
                  "tracks": {k: {f: v[f] for f in ("cases", "questions", "coverage", "accuracy", "case_exact_accuracy")}
                             for k, v in result["tracks"].items()}}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
