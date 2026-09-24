import collections
import gzip
import hashlib
import json
import shutil
from pathlib import Path

from decision_index import editions
from decision_index.suite.build import adapters_added


def excluded_groups(rows_path, excluded):
    groups = collections.defaultdict(set)
    with open(rows_path, encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)["_evaluation"]
            if e["run_id"] in excluded:
                groups[e["catalog_id"]].add(e["group_id"])
    return groups


def cut(v1_rows, out_path, excluded):
    subsets = editions.retrieval_subsets()["subsets"]
    keep = {s["catalog_id"]: set(s["group_ids"]) for s in subsets.values()}
    carried = excluded_groups(v1_rows, excluded)
    digest = hashlib.sha256()
    counts = collections.Counter()
    with open(v1_rows, encoding="utf-8") as f, open(out_path, "w", encoding="utf-8", newline="\n") as out:
        for line in f:
            e = json.loads(line)["_evaluation"]
            n = e["catalog_id"]
            if n in keep and e["group_id"] not in keep[n] and e["group_id"] not in carried[n]:
                continue
            out.write(line)
            digest.update(line.encode())
            counts[n] += 1
    return {"requests": sum(counts.values()), "sha256": digest.hexdigest(), "toolret_requests": counts[2], "bright_requests": counts[36]}


def gz(path):
    target = Path(str(path) + ".gz")
    with open(path, "rb") as src, gzip.open(target, "wb", compresslevel=9) as dst:
        shutil.copyfileobj(src, dst)
    return target


def main(layout, v1_rows, exclusions_path, added=None, skip_download=False, log=print):
    e = editions.get("0.2")
    out = layout.suite / "release-v2-rebuilt"
    out.mkdir(parents=True, exist_ok=True)
    result = {"out": str(out)}
    if v1_rows is not None:
        excluded = set(json.loads(Path(exclusions_path).read_text())["rows"])
        rows = out / "selected-rows.jsonl"
        v2 = cut(v1_rows, rows, excluded)
        gz(rows)
        result.update(rows=v2, rows_byte_identical=v2["sha256"] == e["rows_sha256"])
        log(json.dumps({"event": "release_v2_cut", **v2, "byte_identical": result["rows_byte_identical"]}))
    numbers = list(adapters_added.ORDER) if added is None else [n for n in adapters_added.ORDER if n in added]
    if not numbers:
        return result
    if not skip_download:
        adapters_added.acquire(layout, numbers, log=log)
    added_path = out / "added-rows.jsonl"
    added = adapters_added.build(layout, added_path, numbers, log=log)
    gz(added_path)
    result.update(added=added, added_byte_identical=added["sha256"] == e["added_sha256"])
    return result
