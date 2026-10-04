"""Single claimed ESDB final test; requires stage-test from a certified protocol."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if ROOT == Path("/"):
    ROOT = Path("/root")
SEV = Path(os.environ.get("ESDB_SEV_ROOT", str(ROOT.parent / "Sev-security-20260925"))).resolve() if ROOT != Path("/root") else Path("/root")
os.environ.setdefault("KEV_APP_NAME", "sev-esdb-v03-final")
sys.path.insert(0, str(SEV))
import modal_app
import modal

approvals = modal.Dict.from_name("esdb-certified-final-approvals", create_if_missing=True)
claims = modal.Dict.from_name("esdb-final-execution-claims", create_if_missing=True)

INPUT = ROOT / "work/esdb-sev-v03-test.jsonl.gz" if ROOT != Path("/root") else ROOT / "esdb-test.jsonl.gz"
PLAN = INPUT.with_suffix(".plan.json") if ROOT != Path("/root") else ROOT / "esdb-test-plan.json"
app = modal_app.app
image = (modal_app.image.add_local_dir(ROOT / "decision_index", "/root/decision_index")
         .add_local_file(INPUT, "/root/esdb-test.jsonl.gz")
         .add_local_file(PLAN, "/root/esdb-test-plan.json")
         .add_local_file(SEV / "modal_app.py", "/root/modal_app.py"))


@app.function(image=image, gpu="H100", cpu=2, memory=(32768, 131072), retries=0, timeout=3600,
              volumes={"/runs": modal_app.runs_volume, "/hf": modal_app.hf_cache}, secrets=modal_app.secrets)
def final(name, expected_plan_hash):
    from decision_index.cyber.common import file_hash, read_json, rows
    from decision_index.cyber.v3.inference import _evaluate as evaluate
    from decision_index.cyber.v3.protocol import check_pipeline_code
    plan = read_json("/root/esdb-test-plan.json")
    if (file_hash("/root/esdb-test-plan.json") != expected_plan_hash or not plan["review_certified"]
            or file_hash("/root/esdb-test.jsonl.gz") != plan["input_sha256"]):
        raise ValueError("Final test plan or requests changed")
    if any(r["split"] != "test" for r in rows("/root/esdb-test.jsonl.gz")):
        raise ValueError("Final test input contains another partition")
    if approvals.get(plan["protocol_sha256"]) != expected_plan_hash:
        raise ValueError("Final plan was not approved by the sealed local review protocol")
    check_pipeline_code(plan["pipeline_code"])
    if not claims.put(plan["protocol_sha256"], expected_plan_hash, skip_if_exists=True):
        raise FileExistsError("Final inference already claimed across workers")
    # The immutable remote path follows the protocol hash, not a chosen run
    # name. Changing the name cannot start another inference on this protocol.
    remote = f"/runs/esdb-final/{plan['protocol_sha256']}"
    Path(remote).parent.mkdir(parents=True, exist_ok=True)
    claim = Path(remote + ".claimed")
    with claim.open("x") as f:
        f.write(expected_plan_hash)
    modal_app.runs_volume.commit()
    result = evaluate("/root/esdb-test.jsonl.gz", remote)
    modal_app.runs_volume.commit()
    return {**result, "remote": remote}


@app.local_entrypoint()
def run(name: str = "sev-v03-final"):
    from decision_index.cyber.common import file_hash, read_json
    from decision_index.cyber.v3.protocol import check_frozen_protocol
    plan = read_json(PLAN)
    protocol = check_frozen_protocol(plan["protocol"])
    dataset = Path(protocol["dataset"])
    claim = read_json(dataset / ".final-test-started.json")
    if claim["protocol_sha256"] != plan["protocol_sha256"] or file_hash(plan["protocol"]) != plan["protocol_sha256"]:
        raise ValueError("Final test has no matching local execution claim")
    if not approvals.put(plan["protocol_sha256"], file_hash(PLAN), skip_if_exists=True) and approvals.get(plan["protocol_sha256"]) != file_hash(PLAN):
        raise ValueError("A different final plan was registered for this protocol")
    print("One certified final test, maximum one H100 hour, compute upper bound $5.07.")
    result = final.remote(name, file_hash(PLAN))
    modal_app.pull_volume(result["remote"], ROOT / "runs")
    print(result)
