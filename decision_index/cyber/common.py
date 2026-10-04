import gzip
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

VERSION = "esdb-v0.1"
SEED = "enterprise-security-decisions-20261003-v1"
SPLITS = ("development", "calibration", "test")
TRACKS = {"incident_triage": 1001, "endpoint_investigation": 1002,
          "network_defense": 1003, "authorization_policy": 1004}


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(encoded(value).encode("utf-8")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rank(*values):
    return digest([SEED, *values])


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_rows(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        if path.suffix == ".gz":
            with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as zipped:
                with io.TextIOWrapper(zipped, encoding="utf-8", newline="\n") as f:
                    for row in values:
                        f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
        else:
            for row in values:
                raw.write((json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8"))


def receipt(path, root, role="source"):
    path, root = Path(path), Path(root)
    return {"path": str(path.relative_to(root)), "root": role,
            "sha256": file_hash(path), "bytes": path.stat().st_size}


def assign_groups(groups, namespace):
    """Assign entire independent groups, without looking at answers or scores."""
    ordered = sorted(set(groups), key=lambda g: (rank("split", namespace, g), g))
    if len(ordered) < 3:
        raise ValueError(f"{namespace}: need at least three independent groups")
    return {g: SPLITS[i % 3] for i, g in enumerate(ordered)}


def choice(case, field, instructions, descriptions, answer):
    if answer not in descriptions or not 2 <= len(descriptions) <= 255:
        raise ValueError("answer missing from supplied choices")
    keys = sorted(descriptions, key=lambda k: rank("options", case, field, k))
    return {"type": "choice", "instructions": instructions,
            "criteria": {k: descriptions[k] for k in keys}}, answer


def record(track, source_id, split, split_group, state, questions, expected, metadata, provenance):
    if split not in SPLITS or set(questions) != set(expected):
        raise ValueError("invalid record partition or answer fields")
    rid = f"{VERSION}:{track}:{digest(source_id)}"
    payload = {"state": state, "questions": questions}
    return {"id": digest(source_id), "family": track, "split": split,
            **payload, "expected": expected,
            "metadata": {"split_group": split_group, **metadata}, "provenance": provenance,
            "_evaluation": {"run_id": rid, "catalog_id": TRACKS[track], "dataset": track,
                            "group_id": digest(source_id), "track": metadata.get("task", track),
                            "source_path": provenance.get("path", ""), "payload_sha256": digest(payload),
                            "proxy_tokens": None, "benchmark_origin": "Enterprise Security Decision Benchmark"}}


def summary(values):
    return {"records": len(values), "questions": sum(len(r["questions"]) for r in values),
            "independent_groups": len({r["metadata"]["split_group"] for r in values}),
            "labels": dict(sorted(Counter(
                r["questions"][q]["criteria"][gold] for r in values for q, gold in r["expected"].items()
            ).items()))}
