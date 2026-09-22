import collections
import gzip
import hashlib

from decision_index.suite.io import dumps


def stratified_sample(suite, n, out_path, seed=20260919, apply_exclusions=True, complete_groups=True):
    groups = collections.defaultdict(list)
    for r in suite.rows(apply_exclusions=apply_exclusions):
        e = r["_evaluation"]
        groups[(e["catalog_id"], e["group_id"])].append(r)
    by_benchmark = collections.defaultdict(list)
    for key, rows in groups.items():
        by_benchmark[key[0]].append(rows)
    benchmarks = sorted(by_benchmark)
    ranked = {b: sorted(by_benchmark[b], key=lambda rows: hashlib.sha256(f"{seed}:{b}:{rows[0]['_evaluation']['group_id']}".encode()).hexdigest()) for b in benchmarks}
    chosen = []
    cursor = {b: 0 for b in benchmarks}
    total = 0
    while total < n:
        progressed = False
        for b in benchmarks:
            if total >= n:
                break
            if cursor[b] < len(ranked[b]):
                rows = ranked[b][cursor[b]]
                cursor[b] += 1
                progressed = True
                take = rows if complete_groups else rows[:1]
                chosen.extend(take)
                total += len(take)
        if not progressed:
            break
    opener = gzip.open if str(out_path).endswith(".gz") else open
    with opener(out_path, "wt", encoding="utf-8") as f:
        for r in chosen:
            f.write(dumps(r) + "\n")
    return {"rows": len(chosen), "benchmarks": len(benchmarks), "path": str(out_path)}
