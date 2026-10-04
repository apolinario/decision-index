"""Collect known Sev use from non-test partitions; never read existing tests."""
from pathlib import Path

from .common import digest, file_hash, read_json, receipt, rows

GUIDE_SEED = "sev-guide-pilot-v1-20260926"


def legacy_guide_group(org):
    return digest([GUIDE_SEED, "organization", org])


def collect(collection, sev_root):
    collection, sev_root = Path(collection), Path(sev_root)
    result = {k: set() for k in ("groups", "states", "archives", "native_lines", "native_events",
                                "program_ids", "program_texts", "templates", "root_ids")}
    result["receipts"] = []
    directories = [(p, collection, "collection") for parent in ("datasets", "external")
                   for p in sorted((collection / parent).iterdir())
                   if p.is_dir() and p.name.startswith(("sev_", "sev-"))]
    directories += [(p, sev_root, "sev") for p in sorted((sev_root / "evals/sev").iterdir()) if p.is_dir()]
    for directory, root, role in directories:
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = read_json(manifest_path)
        for split in ("train", "calibration", "development"):
            path = directory / f"{split}.jsonl"
            if not path.exists():
                continue
            pin = manifest.get("files", {}).get(path.name, {})
            actual = file_hash(path)
            if isinstance(pin, dict) and pin.get("sha256") and pin["sha256"] != actual:
                raise ValueError(f"protected partition differs from its manifest: {path}")
            count = 0
            for row in rows(path):
                count += 1
                meta = row.get("_meta", row.get("metadata", {}))
                if meta.get("group_id"):
                    result["groups"].add(meta["group_id"])
                if "state" in row:
                    result["states"].add(digest(row["state"]))
                for key, target in (("source_record_id", "program_ids"), ("source_text_sha256", "program_texts"),
                                    ("template_sha256", "templates"), ("source_root_id", "root_ids")):
                    if meta.get(key):
                        result[target].add(meta[key])
                for ref in meta.get("native_refs", []):
                    for key, target in (("archive_sha256", "archives"), ("native_line_sha256", "native_lines"),
                                        ("event_identity", "native_events")):
                        if ref.get(key):
                            result[target].add(ref[key])
            result["receipts"].append({**receipt(path, root, role), "records": count,
                                       "manifest_sha256": file_hash(manifest_path)})
    # Metadata-only manual protection ledger does not reveal locked test payloads.
    manual = collection / "external/sev-swarmtraces-manual-v1/protection.json"
    if manual.exists():
        x = read_json(manual)
        result["root_ids"].update(x.get("exact_text_ancestry_closure_root_ids", []))
        result["receipts"].append(receipt(manual, collection))
    return result


def export(result):
    return {k: sorted(v) if isinstance(v, set) else v for k, v in result.items()}
