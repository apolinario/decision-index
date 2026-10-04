"""Independent review certificates, calibration proposals and a one-run test lock."""
import math
import os
from collections import defaultdict
from pathlib import Path

from ..build import load_partition
from ..common import digest, file_hash, read_json, rows, write_json
from .build import verify
from .common import REVISION
from .inference import CONTEXT
from .scoring import score


def certify(dataset, profiles, annotations, out, packet_path=None):
    verify(dataset)
    dataset, out = Path(dataset), Path(out)
    if out.exists():
        raise FileExistsError("Review certificates are immutable")
    if len(profiles) != 2 or len(annotations) != 2:
        raise ValueError("Two independent reviewer profiles and annotations required")
    required = set(read_json(dataset / "review/requirements.json")["required_case_ids"])
    values = {r["id"]: r for split in ("development", "calibration", "test") for r in load_partition(dataset, split)}
    packet_path = Path(packet_path or dataset / "review/blinded-cases.jsonl.gz")
    packet = list(rows(packet_path))
    if len(packet) != len(required) or {r["id"] for r in packet} != required:
        raise ValueError("Blinded packet does not cover every required case exactly once")
    allowed_source = {"path", "sha256", "native_refs", "source_text_sha256", "source_line", "source_line_sha256", "byte_start", "byte_end"}
    for item in packet:
        if set(item) != {"id", "track", "state", "questions", "source_evidence"} or set(item["source_evidence"]) - allowed_source:
            raise ValueError("Blinded review packet exposes generated annotations or unsupported provenance")
        row = values[item["id"]]
        if item["state"] != row["state"] or item["questions"] != row["questions"]:
            raise ValueError("Review evidence differs from the frozen model evidence")
    identities, names, receipts, disagreements = set(), set(), [], []
    for profile_path, annotation_path in zip(profiles, annotations):
        profile = read_json(profile_path)
        if (not all(profile.get(k) for k in ("reviewer_id", "name", "security_expertise"))
                or profile.get("independent_review") is not True or "conflicts" not in profile):
            raise ValueError("Reviewer must identify security expertise, conflicts and independent completion")
        rid = profile["reviewer_id"]
        if rid in identities or profile["name"] in names:
            raise ValueError("Reviewers must be distinct people")
        identities.add(rid)
        names.add(profile["name"])
        answers = list(rows(annotation_path))
        if len(answers) != len(required) or {r["id"] for r in answers} != required:
            raise ValueError("Reviewer did not complete every required case exactly once")
        for answer in answers:
            if answer.get("reviewer_id") != rid or set(answer.get("answers", {})) != set(values[answer["id"]]["expected"]):
                raise ValueError("Reviewer answer identity or field coverage invalid")
            if answer["answers"] != values[answer["id"]]["expected"] or (answer.get("issue", "") or "").strip():
                disagreements.append({"reviewer_id": rid, "id": answer["id"], "answers": answer["answers"], "issue": answer.get("issue")})
        receipts.append({"profile": {"path": str(Path(profile_path).resolve()), "sha256": file_hash(profile_path)},
                         "annotations": {"path": str(Path(annotation_path).resolve()), "sha256": file_hash(annotation_path)}, "reviewer_id": rid})
    if disagreements:
        write_json(out.with_suffix(".disagreements.json"), {"certified": False, "issues": disagreements,
            "next_step": "Independent adjudication required. Changed gold must ship as a new immutable dataset edition before test."})
        raise ValueError(f"{len(disagreements)} reviewer disagreements/issues require adjudication; no certificate issued")
    certificate = {"certified": True, "dataset_manifest_sha256": file_hash(dataset / "manifest.json"),
        "blinded_packet": str(packet_path.resolve()), "blinded_packet_sha256": file_hash(packet_path), "reviewed_cases": len(required), "reviewers": receipts,
        "basis": "Two identified security reviewers' independent completed annotations; no generated or model-only review substituted."}
    write_json(out, certificate)
    return certificate


def softmax(logits, temperature):
    top = max(logits.values())
    exps = {k: math.exp((z - top) / temperature) for k, z in logits.items()}
    total = sum(exps.values())
    return {k: v / total for k, v in exps.items()}


def check_review_certificate(certificate_path, dataset):
    """Bind the certification to the reviewed packet and unchanged reviewers."""
    certificate = read_json(certificate_path)
    required = read_json(dataset / "review/requirements.json")["required_case_ids"]
    if (not certificate.get("certified")
            or certificate.get("dataset_manifest_sha256") != file_hash(dataset / "manifest.json")
            or certificate.get("reviewed_cases") != len(required)
            or len(certificate.get("reviewers", [])) != 2):
        raise ValueError("Independent review certificate missing or for a different dataset")
    if file_hash(certificate["blinded_packet"]) != certificate["blinded_packet_sha256"]:
        raise ValueError("Blinded review packet changed after certification")
    for reviewer in certificate["reviewers"]:
        for role in ("profile", "annotations"):
            if file_hash(reviewer[role]["path"]) != reviewer[role]["sha256"]:
                raise ValueError("Reviewer receipt changed")
    return certificate


def pipeline_pins():
    root = Path(__file__).resolve().parents[2]
    files = [*root.glob("cyber/*.py"), *root.glob("cyber/v3/*.py"), *root.glob("cyber/v3/vendor/*"), root / "engines/base.py", root / "results.py", root / "suite/io.py"]
    return {str(p.relative_to(root)): file_hash(p) for p in sorted(files)}


def check_pipeline_code(pins, root=None):
    root = Path(root or Path(__file__).resolve().parents[2])
    for name, pin in pins.items():
        if file_hash(root / name) != pin:
            raise ValueError("Evaluation code changed after protocol freeze")


def calibrate(values, results, out):
    from scipy.optimize import minimize_scalar
    by_run = {r["run_id"]: r for r in results}
    if len(results) != len(values) or set(by_run) != {r["_evaluation"]["run_id"] for r in values}:
        raise ValueError("Calibration must contain one attempt for every frozen request")
    grouped, temperatures = defaultdict(list), {}
    for row in values:
        result = by_run[row["_evaluation"]["run_id"]]
        if result["payload_sha256"] != row["_evaluation"]["payload_sha256"]:
            raise ValueError("Calibration result is for different model evidence")
        if result["status"] != "ok" or row["metadata"].get("diagnostic_only", False):
            continue
        for field, gold in row["expected"].items():
            grouped[row["family"]].append((result["logits"][field], gold, row["metadata"]["split_group"]))
    for track, items in grouped.items():
        group_sizes = defaultdict(int)
        for _, _, group in items:
            group_sizes[group] += 1
        def loss(log_t):
            total = sum(-math.log(max(softmax(z, math.exp(log_t))[y], 1e-15)) / group_sizes[g] for z, y, g in items)
            return total / len(group_sizes)
        fitted = minimize_scalar(loss, bounds=(math.log(.1), math.log(10)), method="bounded", options={"xatol": 1e-5})
        temperatures[track] = {"temperature": math.exp(fitted.x), "group_weighted_nll_before": loss(0),
                               "group_weighted_nll_after": fitted.fun, "groups": len(group_sizes), "answered_questions": len(items)}
    adjusted = []
    for row in values:
        source = by_run[row["_evaluation"]["run_id"]]
        if source["status"] != "ok":
            adjusted.append(source)
            continue
        answers = {}
        for field in row["questions"]:
            dist = softmax(source["logits"][field], temperatures[row["family"]]["temperature"])
            answers[field] = {"type": "choice", "choice": max(dist, key=dist.get), "probabilities": dist}
        adjusted.append({**source, "response": {"answers": answers}})
    proposal = {"status": "settings proposal; independent review certification required before sealing", "temperatures": temperatures,
        "calibration_requests_sha256": digest(sorted((r["_evaluation"]["run_id"], r["_evaluation"]["payload_sha256"]) for r in values)),
        "calibration_results_sha256": digest(sorted(results, key=lambda r: r["run_id"])),
        "calibration_fit": "Equal source-group weight, bounded scalar temperature per track; non-diagnostic calibration cases only. In-sample fit, not validation proof.",
        "threshold": None, "threshold_note": "No operational acceptance threshold is asserted. Report complete accuracy, coverage, confidence and case-exact metrics.",
        "checkpoint": "macmacmacmac/Sev-4B", "revision": REVISION, "context": CONTEXT, "rotations": 1,
        "rendering": "kev.api.to_record", "date_facts": False, "dtype": "fp32", "backend": "torch", "device": "cuda",
        "calibration_raw_report": score(values, results), "calibration_temperature_report": score(values, adjusted)}
    write_json(out, proposal)
    return proposal


def seal(dataset, certificate_path, proposal_path, model_run, baselines, out):
    verify(dataset)
    dataset, model_run, baselines, out = map(Path, (dataset, model_run, baselines, out))
    if out.exists():
        raise FileExistsError("Frozen protocols cannot be overwritten")
    check_review_certificate(certificate_path, dataset)
    provenance = read_json(model_run / "provenance.json")
    completion = read_json(model_run / "completion.json")
    proposal = read_json(proposal_path)
    if provenance["revision"] != REVISION or not completion.get("complete_attempts"):
        raise ValueError("Development/calibration Sev run did not finish")
    if (completion.get("results_sha256") != file_hash(model_run / "results.jsonl")
            or completion.get("input_sha256") != provenance.get("input_sha256")):
        raise ValueError("Development/calibration completion receipt changed")
    devcal = [r for split in ("development", "calibration") for r in load_partition(dataset, split)]
    results = list(rows(model_run / "results.jsonl"))
    expected_runs = {r["_evaluation"]["run_id"]: r for r in devcal}
    if len(results) != len(devcal) or {r["run_id"] for r in results} != set(expected_runs):
        raise ValueError("Sev run does not exactly cover this development/calibration release")
    if any(r["payload_sha256"] != expected_runs[r["run_id"]]["_evaluation"]["payload_sha256"] for r in results):
        raise ValueError("Sev development/calibration result payload changed")
    calibration = [r for r in devcal if r["split"] == "calibration"]
    calibration_runs = {r["_evaluation"]["run_id"] for r in calibration}
    if (proposal["calibration_requests_sha256"] != digest(sorted((r["_evaluation"]["run_id"], r["_evaluation"]["payload_sha256"]) for r in calibration))
            or proposal["calibration_results_sha256"] != digest(sorted((r for r in results if r["run_id"] in calibration_runs), key=lambda r: r["run_id"]))):
        raise ValueError("Calibration settings do not bind to this frozen release and model run")
    protocol = {"sealed": True, "dataset": str(dataset.resolve()), "dataset_manifest_sha256": file_hash(dataset / "manifest.json"),
        "review_certificate": {"path": str(Path(certificate_path).resolve()), "sha256": file_hash(certificate_path)},
        "settings": {k: proposal[k] for k in ("temperatures", "threshold", "checkpoint", "revision", "context", "rotations", "rendering", "date_facts", "dtype", "backend", "device")},
        "model_provenance": provenance, "calibration_proposal_sha256": file_hash(proposal_path),
        "development_calibration_result_sha256": file_hash(model_run / "results.jsonl"),
        "baseline_models": {"directory": str((baselines / "models").resolve()), "recipes_sha256": file_hash(baselines / "models/recipes.json"),
                            "files": {p.name: file_hash(p) for p in (baselines / "models").glob("*.joblib")}},
        "pipeline_code": pipeline_pins(),
        "final_test": "One atomic execution claim; all outcomes retained; no retries of successful predictions or post-test setting selection."}
    write_json(out, protocol)
    return protocol


def check_frozen_protocol(protocol_path):
    protocol = read_json(protocol_path)
    if not protocol.get("sealed"):
        raise ValueError("Final test requires a sealed protocol")
    dataset = Path(protocol["dataset"])
    verify(dataset)
    if file_hash(dataset / "manifest.json") != protocol["dataset_manifest_sha256"]:
        raise ValueError("Dataset changed after protocol freeze")
    cert = protocol["review_certificate"]
    if file_hash(cert["path"]) != cert["sha256"]:
        raise ValueError("Review certificate changed")
    check_review_certificate(cert["path"], dataset)
    check_pipeline_code(protocol["pipeline_code"])
    baseline_models = protocol.get("baseline_models")
    if baseline_models:
        model_dir = Path(baseline_models["directory"])
        if file_hash(model_dir / "recipes.json") != baseline_models["recipes_sha256"]:
            raise ValueError("Baseline recipe changed after freeze")
        for name, pin in baseline_models["files"].items():
            if file_hash(model_dir / name) != pin:
                raise ValueError("Baseline weights changed after freeze")
    return protocol


def claim_test(protocol_path, out):
    protocol = check_frozen_protocol(protocol_path)
    dataset = Path(protocol["dataset"])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # The lock belongs to the dataset, not a user-chosen output name, so choosing
    # a new directory cannot accidentally run this test again.
    lock = dataset / ".final-test-started.json"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as f:
        import json
        json.dump({"protocol_sha256": file_hash(protocol_path), "output": str(out.resolve()), "state": "claimed_before_test_input_open"}, f)
    return protocol
