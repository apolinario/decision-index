"""Check a run for "next option" bias on long option lists.

For every scored single-answer `choice` request with at least --min-options options, compare
the predicted option's position in `criteria` with the gold option's: a model that drifts to
the option listed right after the right one shows a gold+1 rate well above its gold-1 rate,
which serves as the control. See docs/engines.md, "Option pooling in causal towers".

  python scripts/check_next_option_bias.py runs/NAME/results.jsonl[.gz] [--suite-dir suite-0.2]

Standard library only. Reads the suite rows for `expected`, joined on run_id.
"""
import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path


def lines(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("results")
    ap.add_argument("--suite-dir", default="suite-0.2")
    ap.add_argument("--min-options", type=int, default=20)
    a = ap.parse_args()

    gold = {}
    for name in ("selected-rows.jsonl.gz", "added-rows.jsonl.gz"):
        p = Path(a.suite_dir) / name
        if not p.exists():
            continue
        for row in lines(p):
            qs = row.get("questions") or {}
            exp = row.get("expected") or {}
            if len(qs) != 1:
                continue
            (key, q), = qs.items()
            g = exp.get(key)
            if q.get("type") == "choice" and isinstance(g, str) and len(q.get("criteria") or {}) >= a.min_options:
                gold[row["_evaluation"]["run_id"]] = (key, list(q["criteria"]), g)

    stats = defaultdict(lambda: [0, 0, 0, 0])          # n, correct, gold+1, gold-1
    for r in lines(a.results):
        hit = gold.get(r.get("run_id"))
        if hit is None or r.get("status") != "ok":
            continue
        key, order, g = hit
        pred = ((r.get("response") or {}).get("answers") or {}).get(key, {}).get("choice")
        if pred not in order or g not in order:
            continue
        d = order.index(pred) - order.index(g)
        s = stats[r.get("track") or r.get("dataset")]
        s[0] += 1
        s[1] += d == 0
        s[2] += d == 1
        s[3] += d == -1

    if not stats:
        raise SystemExit("no scored choice requests with that many options (check --suite-dir)")
    print(f"{'benchmark':<34}{'n':>7}{'acc':>8}{'gold+1':>9}{'gold-1':>9}")
    for name, (n, c, nx, pv) in sorted(stats.items(), key=lambda kv: -kv[1][2] / kv[1][0]):
        print(f"{name:<34}{n:>7}{c / n:>8.3f}{nx / n:>9.3f}{pv / n:>9.3f}")


if __name__ == "__main__":
    main()
