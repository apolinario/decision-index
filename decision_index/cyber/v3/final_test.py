"""Final evaluation orchestration: claim before opening test, then retain all outcomes."""
from pathlib import Path
from ..gate import final_scoring

from ..build import load_partition
from ..common import file_hash, read_json, rows, write_rows, write_json
from ..scoring import baseline
from .protocol import claim_test, check_frozen_protocol, softmax
from .baselines import classify, deterministic
from .scoring import score


@final_scoring
def stage(protocol_path, out, input_path):
    import joblib
    if Path(input_path).exists():
        raise FileExistsError("Final test staging input already exists")
    protocol = claim_test(protocol_path, out)
    dataset, out, input_path = Path(protocol["dataset"]), Path(out), Path(input_path)
    model_dir = Path(protocol["baseline_models"]["directory"])
    for name, pin in protocol["baseline_models"]["files"].items():
        if file_hash(model_dir / name) != pin:
            raise ValueError("Baseline weights changed after freeze")
    values = load_partition(dataset, "test")
    development = load_partition(dataset, "development")
    models = {Path(name).stem: joblib.load(model_dir / name) for name in protocol["baseline_models"]["files"]}
    engines = {mode: baseline(values, development, mode) for mode in ("uniform", "development_majority")}
    reference = [r for r in values if r["family"] in ("authorization_policy", "endpoint_investigation") or r["metadata"]["task"] == "native_network_boundary_rule"]
    engines["visible_rule_reference"] = [deterministic(r) for r in reference]
    engines.update({name: classify(values, models, name) for name in models})
    for name, predictions in engines.items():
        runs = {p["run_id"] for p in predictions}
        subset = [r for r in values if r["_evaluation"]["run_id"] in runs]
        write_rows(out / f"{name}.jsonl", predictions)
        write_json(out / f"{name}.json", score(subset, predictions))
    write_rows(input_path, list(rows(dataset / "inputs/test.jsonl.gz")))
    plan = {"protocol_sha256": file_hash(protocol_path), "input_sha256": file_hash(input_path), "cases": len(values),
            "dataset_manifest_sha256": protocol["dataset_manifest_sha256"], "review_certified": True,
            "protocol": str(Path(protocol_path).resolve()), "model_provenance": protocol["model_provenance"],
            "pipeline_code": protocol["pipeline_code"],
            "claimed_test_output": str(out.resolve()), "gold_exposed": False}
    write_json(Path(input_path).with_suffix(".plan.json"), plan)
    return {"test_claimed": True, "input": str(input_path), "cases": len(values), "baseline_output": str(out)}


@final_scoring
def finish(protocol_path, model_run, out):
    protocol = check_frozen_protocol(protocol_path)
    dataset, model_run, out = Path(protocol["dataset"]), Path(model_run), Path(out)
    lock = read_json(dataset / ".final-test-started.json")
    if lock["protocol_sha256"] != file_hash(protocol_path) or lock["output"] != str(out.resolve()):
        raise ValueError("Final output does not match the one claimed test")
    if (out / "Sev-report.json").exists():
        raise FileExistsError("Final test has already been scored")
    actual = read_json(model_run / "provenance.json")
    completion = read_json(model_run / "completion.json")
    manifest = read_json(dataset / "manifest.json")
    if (not completion.get("complete_attempts")
            or completion.get("results_sha256") != file_hash(model_run / "results.jsonl")
            or actual["input_sha256"] != manifest["files"]["inputs/test.jsonl.gz"]["sha256"]
            or completion.get("input_sha256") != actual["input_sha256"]):
        raise ValueError("Final inference completion does not bind to frozen test input and results")
    for key in ("revision", "base_revision", "adapter_sha256", "head_sha256", "backend", "dtype", "context", "api_sha256", "model_code_sha256", "checkpoint_code_sha256"):
        if actual[key] != protocol["model_provenance"][key]:
            raise ValueError(f"Final model differs from frozen calibration model: {key}")
    values = load_partition(dataset, "test")
    raw = list(rows(model_run / "results.jsonl"))
    if len(raw) != len(values) or {r["run_id"] for r in raw} != {r["_evaluation"]["run_id"] for r in values}:
        raise ValueError("Final test is incomplete or contains repeated prediction attempts")
    by_run = {r["_evaluation"]["run_id"]: r for r in values}
    calibrated = []
    for result in raw:
        if result["status"] != "ok":
            calibrated.append(result)
            continue
        row = by_run[result["run_id"]]
        temperature = protocol["settings"]["temperatures"][row["family"]]["temperature"]
        answers = {}
        for field, logits in result["logits"].items():
            distribution = softmax(logits, temperature)
            answers[field] = {"type": "choice", "choice": max(distribution, key=distribution.get), "probabilities": distribution}
        calibrated.append({**result, "response": {"answers": answers}})
    write_json(out / "Sev-raw-report.json", score(values, raw))
    write_json(out / "Sev-report.json", score(values, calibrated))
    write_rows(out / "Sev-calibrated.jsonl", calibrated)
    write_json(out / "completion.json", {"final_test_complete": True, "protocol_sha256": file_hash(protocol_path),
        "model_results_sha256": file_hash(model_run / "results.jsonl"), "model_attempts": len(raw), "repeated_predictions": False})
    return {"final_test_complete": True, "report": str(out / "Sev-report.json")}
