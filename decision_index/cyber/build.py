"""Resumable local construction and immutable, answer-isolated local files."""
import os
import tempfile
from pathlib import Path

from . import endpoint, guide, network, policy, protection
from .audit import deduplicate, replay, validate
from .common import SEED, SPLITS, TRACKS, VERSION, digest, file_hash, read_json, rows, summary, write_json, write_rows
from .scoring import baseline, score


def code_receipts():
    return {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def source_receipts(value):
    found = {}

    def visit(x):
        if isinstance(x, dict):
            if "path" in x and "sha256" in x:
                key = x.get("root", "collection"), x["path"]
                found[key] = {"root": key[0], "path": key[1], "sha256": x["sha256"]}
            for v in x.values():
                visit(v)
        elif isinstance(x, list):
            for v in x:
                visit(v)
    visit(value)
    return sorted(found.values(), key=lambda r: (r["root"], r["path"]))


def verify_sources(receipts, collection, sev_root):
    for r in receipts:
        root = sev_root if r["root"] == "sev" else collection
        if file_hash(root / r["path"]) != r["sha256"]:
            raise ValueError(f"source receipt mismatch: {r['path']}")


def load_partition(directory, split):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    inputs = list(rows(directory / f"inputs/{split}.jsonl.gz"))
    labels = list(rows(directory / f"labels/{split}.jsonl.gz"))
    for relative in (f"inputs/{split}.jsonl.gz", f"labels/{split}.jsonl.gz"):
        if file_hash(directory / relative) != manifest["files"][relative]["sha256"]:
            raise ValueError("frozen partition hash mismatch")
    by_id = {r["id"]: r for r in labels}
    if len(by_id) != len(labels) or len({r["id"] for r in inputs}) != len(inputs) or set(by_id) != {r["id"] for r in inputs}:
        raise ValueError("input/answer partition identities do not match uniquely")
    combined = [{**r, **by_id[r["id"]]} for r in inputs]
    if any(r["split"] != split for r in combined):
        raise ValueError("partition contains a different split")
    return combined


def verify(directory):
    directory = Path(directory)
    m = read_json(directory / "manifest.json")
    for relative, r in m["files"].items():
        path = directory / relative
        if path.stat().st_size != r["bytes"] or file_hash(path) != r["sha256"]:
            raise ValueError(f"frozen artifact mismatch: {relative}")
    values = [r for split in SPLITS for r in load_partition(directory, split)]
    p = read_json(directory / "reports/protection.json")
    p = {k: set(v) if k != "receipts" else v for k, v in p.items()}
    a = validate(values, p)
    if a != read_json(directory / "reports/audit.json")["structural"]:
        raise ValueError("frozen audit no longer matches dataset")
    if m["counts"] != {track: summary([r for r in values if r["family"] == track]) for track in TRACKS}:
        raise ValueError("frozen manifest counts no longer match dataset")
    return {"version": m["version"], "files_verified": len(m["files"]),
            "records": a["total"]["records"], "questions": a["total"]["questions"],
            "independent_groups": a["total"]["independent_groups"], "checks": a["checks"]}


def build(collection, sev_root, out, work, *, typescript=None, log=print):
    collection, sev_root, out, work = map(lambda p: Path(p).resolve(), (collection, sev_root, out, work))
    if out.exists():
        raise FileExistsError(f"Frozen destination already exists: {out}; choose a new destination/version.")
    work.mkdir(parents=True, exist_ok=True)
    p = protection.collect(collection, sev_root)
    exported = protection.export(p)
    write_json(work / "protection.json", exported)
    code = code_receipts()
    phase_key = digest({"code": code, "protection": exported, "seed": SEED})
    builders = {"incident_triage": lambda: guide.build(collection, p, log=log),
                "endpoint_investigation": lambda: endpoint.build(collection, p, log=log),
                "network_defense": lambda: network.build(collection, p, log=log),
                "authorization_policy": lambda: policy.build(collection, sev_root, p, log=log)}
    values, reports = [], {}
    for track, builder in builders.items():
        data, report_path, stamp_path = (work / f"{track}.jsonl.gz", work / f"{track}.report.json", work / f"{track}.cache.json")
        cached = False
        if stamp_path.exists() and data.exists() and report_path.exists():
            stamp = read_json(stamp_path)
            cached = (stamp.get("phase_key") == phase_key and file_hash(data) == stamp.get("data_sha256")
                      and file_hash(report_path) == stamp.get("report_sha256"))
            if cached:
                report = read_json(report_path)
                verify_sources(source_receipts(report), collection, sev_root)
                selected = list(rows(data))
                log(f"Verified phase cache: {track}, {len(selected)} cases")
        if not cached:
            selected, report = builder()
            write_rows(data, selected)
            write_json(report_path, report)
            write_json(stamp_path, {"phase_key": phase_key, "data_sha256": file_hash(data), "report_sha256": file_hash(report_path)})
        reports[track] = report
        values.extend(selected)
    values, duplicates = deduplicate(values)
    structural = validate(values, p)
    source_replay = replay(values, collection, sev_root, typescript=typescript, log=log)
    receipts = source_receipts(reports) + source_receipts(exported["receipts"])
    verify_sources(receipts, collection, sev_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".esdb-build-", dir=out.parent) as temp:
        staging = Path(temp) / VERSION
        staging.mkdir()
        for split in SPLITS:
            partition = [r for r in values if r["split"] == split]
            # The runner receives this file. It contains no gold, source provenance or adjudication.
            write_rows(staging / f"inputs/{split}.jsonl.gz", [
                {k: r[k] for k in ("id", "family", "split", "state", "questions", "_evaluation")} for r in partition])
            write_rows(staging / f"labels/{split}.jsonl.gz", [
                {k: r[k] for k in ("id", "expected", "metadata", "provenance")} for r in partition])
        write_json(staging / "reports/protection.json", exported)
        for track, report in reports.items():
            write_json(staging / f"reports/{track}.json", report)
        write_json(staging / "reports/audit.json", {"structural": structural, "deduplication": duplicates, "source_replay": source_replay})
        write_json(staging / "reports/source-receipts.json", receipts)
        development = [r for r in values if r["split"] == "development"]
        calibration = [r for r in values if r["split"] == "calibration"]
        baseline_reports = {mode: score(calibration, baseline(calibration, development, mode))
                            for mode in ("uniform", "development_majority")}
        write_json(staging / "reports/calibration-baselines.json", {
            "scope": "Predeclared baselines on calibration only; majority parameters fitted on development only. "
                     "No model inference or test-set model selection.", "baselines": baseline_reports})
        lines = ["# Enterprise Security Decision Benchmark v0.1", "", "Local research dataset built from ../jevalin-collect.", "",
                 "| Track | Cases | Questions | Source groups |", "|---|---:|---:|---:|"]
        for track in TRACKS:
            counts = summary([r for r in values if r["family"] == track])
            lines.append(f"| {track} | {counts['records']} | {counts['questions']} | {counts['independent_groups']} |")
        lines += ["", "Use inputs/development.jsonl.gz for development, inputs/calibration.jsonl.gz for choosing and freezing thresholds, "
                  "and inputs/test.jsonl.gz for a final locked evaluation. Answer keys, source IDs, and evidence spans are in labels/ only.", "",
                  "The existing Decision Index runner accepts inputs/*.jsonl.gz. Score with python -m decision_index.cyber score. "
                  "This research suite is separate from the published Decision Index 0.2.1.", "",
                  "All selected records were replayed against original sources. Policy annotations were statically parsed; "
                  "human adjudication remains outstanding. Three network capture groups limit generalization. Code-execution "
                  "and console-output policies have no positive examples, nor does the onward-request-body policy. Source terms apply; recovered program reuse terms "
                  "remain unresolved, so this build is not a public redistribution grant.", "",
                  "See ../../docs/cyber-benchmark.md and ../../docs/assessment-review.md for the recipe, assessment review, and acceptance limits.", ""]
        (staging / "README.md").write_text("\n".join(lines), encoding="utf-8")
        files = {str(path.relative_to(staging)): {"sha256": file_hash(path), "bytes": path.stat().st_size}
                 for path in sorted(staging.rglob("*")) if path.is_file()}
        write_json(staging / "manifest.json", {
            "version": VERSION, "seed": SEED, "created_date": "2026-10-03", "status": "source-replayed research benchmark",
            "track_catalog": TRACKS, "builder_sha256": code, "files": files,
            "counts": {track: summary([r for r in values if r["family"] == track]) for track in TRACKS},
            "partitions": structural["partitions"],
            "model_reference": {"repo": "macmacmacmac/Sev-4B", "tag": "v0.3.1-html-contrast-research",
                                "revision": "7f9f3f53b9baae041c64a71d4b3867c6700fd524", "model_inference_performed": False},
            "protection_scope": "Known Sev non-test train/calibration/development partitions and metadata-only component ledgers. "
                                "No existing locked test payloads opened. Public-source pretraining overlap is unknown.",
            "test_policy": "Tune only on development/calibration; freeze checkpoint, rendering, thresholds and budgets before test. "
                           "All questions are data-quality replayed, but test model predictions are not used for selection.",
            "publication_scope": "Local research build; source-specific licenses and unresolved recovered-program terms apply.",
            "limitations": ["No independent expert adjudication or operational field validation.",
                            "GUIDE provider-grade prediction uses anonymized, detector-assisted retrospective evidence.",
                            "Endpoint narrow effects and joins have exact parser baselines; they do not measure open-ended SOC investigation.",
                            "Only three network capture groups; Background is unjudged and binary metrics exclude it.",
                            "Authored policy interpretation is conditional; onward-request-body/code-execution/console policies lack positive examples."]})
        verify(staging)
        if out.exists():
            raise FileExistsError(out)
        os.rename(staging, out)
    log(f"Frozen {structural['total']['records']} cases / {structural['total']['questions']} questions at {out}")
    return verify(out)
