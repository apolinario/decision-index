import collections
import datetime
import hashlib
import json

from decision_index.suite.build.layout import DATASETS, NAMES, REFERENCE_MODEL, STATIC_ORDER
from decision_index.suite.io import dumps, read_jsonl

SEED = 20260919
CASE_CAPS = {22: 2000, 31: 5000, 10: 2500, 23: 5000, 44: 5000, 37: 5000}
REQUEST_CAPS = {6: 10000, 20: 5000}
ORIGINAL = {4, 5, 11, 12, 24, 25, 26, 27, 28, 29, 32, 33, 34, 37, 40, 41, 42, 44, 45}


def provenance(number):
    return "Purpose-built benchmarks" if number == 9 else "Original-task benchmarks" if number in ORIGINAL else "Jev-adapted benchmarks"


def priority(n, g):
    return hashlib.sha256(f"{SEED}:{n}:{g}".encode()).hexdigest()


def gid(r):
    return str(r.get("metadata", {}).get("group_id", r["id"]))


def source_paths(layout, n):
    return [layout.normalized / f"{name}.jsonl" for name in NAMES.get(n, []) if (layout.normalized / f"{name}.jsonl").exists()]


def choose(n, counts, strata):
    chosen = set(counts)
    budget = REQUEST_CAPS.get(n)
    if n == 37:
        buckets = collections.defaultdict(list)
        for g, k in strata.items():
            buckets[k].append(g)
        shares = {k: CASE_CAPS[n] * len(v) / len(counts) for k, v in buckets.items()}
        quota = {k: int(v) for k, v in shares.items()}
        for k in sorted(buckets, key=lambda k: (-(shares[k] - quota[k]), k))[: CASE_CAPS[n] - sum(quota.values())]:
            quota[k] += 1
        chosen = {g for k, gs in buckets.items() for g in sorted(gs, key=lambda g: priority(n, g))[: quota[k]]}
    elif n in CASE_CAPS:
        chosen = set(sorted(counts, key=lambda g: priority(n, g))[: CASE_CAPS[n]])
    elif budget:
        chosen = set()
        used = 0
        for g in sorted(counts, key=lambda g: priority(n, g)):
            if used + counts[g] <= budget:
                chosen.add(g)
                used += counts[g]
    return chosen, budget


def freeze(layout, out_dir, log=print):
    out_dir.mkdir(parents=True, exist_ok=True)
    inventory = []
    selected_paths = []
    active = layout.suite / "active/release-v1"
    active.mkdir(parents=True, exist_ok=True)
    for n in STATIC_ORDER:
        paths = source_paths(layout, n)
        if not paths:
            log(json.dumps({"event": "missing_normalized", "catalog_id": n, "dataset": DATASETS[n]}))
            continue
        counts = collections.Counter()
        strata = {}
        for p in paths:
            for r in read_jsonl(p):
                g = gid(r)
                counts[g] += 1
                if n == 37:
                    strata[g] = (r["metadata"]["locale"], r["expected"]["answer"])
        chosen, budget = choose(n, counts, strata)
        newpaths = []
        req = fields = 0
        for p in paths:
            dst = active / p.name
            with dst.open("w", encoding="utf-8") as f:
                for r in read_jsonl(p):
                    if gid(r) not in chosen:
                        continue
                    f.write(dumps(r) + "\n")
                    req += 1
                    fields += len(r["questions"])
            newpaths.append(dst)
            selected_paths.append((n, DATASETS[n], dst))
        assert not budget or req <= budget
        if n in CASE_CAPS:
            assert len(chosen) == min(CASE_CAPS[n], len(counts))
        entry = dict(catalog_id=n, dataset=DATASETS[n], benchmark_origin=provenance(n), available_cases=len(counts), selected_cases=len(chosen), requests=req, fields=fields, request_cap=budget, case_cap=CASE_CAPS.get(n), selected_group_ids=sorted(chosen), sources=[{"path": layout.rel(p), "sha256": hashlib.file_digest(p.open("rb"), "sha256").hexdigest()} for p in paths])
        inventory.append(entry)
        log(json.dumps({"event": "pool_frozen", **{k: v for k, v in entry.items() if k not in ("selected_group_ids", "sources")}}))
    proxy = 0
    totalfields = 0
    rows_path = out_dir / "selected-rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as sf:
        for n, name, p in selected_paths:
            for r in read_jsonl(p):
                payload = {"model": REFERENCE_MODEL, "state": r["state"], "questions": r["questions"]}
                h = hashlib.sha256(dumps(payload).encode()).hexdigest()
                estimate = max(1, len(dumps(payload)) // 4)
                proxy += estimate
                totalfields += len(r["questions"])
                r["_evaluation"] = dict(run_id=f'{n}:{p.stem}:{r["id"]}', catalog_id=n, dataset=name, group_id=gid(r), track=p.stem, source_path=layout.rel(p), payload_sha256=h, proxy_tokens=estimate, benchmark_origin=provenance(n))
                sf.write(dumps(r) + "\n")
    manifest = dict(model=REFERENCE_MODEL, seed=SEED, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), selection="Deterministic hash sample without replacement; ESCI proportional locale × relevance; request-budget datasets retain complete linked groups; no outcome-based selection.", benchmarks=inventory, static_benchmarks=len(inventory), selected_cases=sum(i["selected_cases"] for i in inventory), requests=sum(i["requests"] for i in inventory), fields=totalfields, selected_rows_sha256=hashlib.file_digest(rows_path.open("rb"), "sha256").hexdigest(), estimated_usd=proxy * 1.925745 * 0.042 / 1e6)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest
