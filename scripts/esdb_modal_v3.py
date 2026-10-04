"""Isolated ESDB evaluation using the Sev repo's canonical Modal app/image.

Inputs must be explicitly staged answer-free development/calibration requests.
Use the v3 CLI's gated final-test staging path for test, never an arbitrary file.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if ROOT == Path("/"):
    ROOT = Path("/root")
SEV = Path(os.environ.get("ESDB_SEV_ROOT", str(ROOT.parent / "Sev-security-20260925"))).resolve() if (ROOT.parent / "Sev-security-20260925").exists() else Path("/root")
os.environ.setdefault("KEV_APP_NAME", "sev-esdb-v03")
sys.path.insert(0, str(SEV))
import modal_app
import modal

INPUT_BASENAME = os.environ.get("ESDB_INPUT_BASENAME", "esdb-sev-v03-devcal.jsonl.gz")
if Path(INPUT_BASENAME).name != INPUT_BASENAME:
    raise ValueError("Input basename must name a staged file inside work/")
INPUT = ROOT / "work" / INPUT_BASENAME if ROOT != Path("/root") else Path("/root/esdb-input.jsonl.gz")
app = modal_app.app
image = (modal_app.image.add_local_dir(ROOT / "decision_index", "/root/decision_index")
         .add_local_file(INPUT, "/root/esdb-input.jsonl.gz")
         .add_local_file(SEV / "modal_app.py", "/root/modal_app.py"))


@app.function(image=image, gpu="H100", cpu=2, memory=(32768, 131072), retries=0, timeout=7200,
              volumes={"/runs": modal_app.runs_volume, "/hf": modal_app.hf_cache}, secrets=modal_app.secrets)
def evaluate(name, expected_input_hash):
    from decision_index.cyber.common import file_hash, rows
    from decision_index.cyber.v3.inference import evaluate as inference
    if file_hash("/root/esdb-input.jsonl.gz") != expected_input_hash:
        raise ValueError("Mounted request bytes differ from staged development/calibration")
    if any(r["split"] not in ("development", "calibration") for r in rows("/root/esdb-input.jsonl.gz")):
        raise ValueError("This entrypoint never admits test")
    result = inference("/root/esdb-input.jsonl.gz", f"/runs/esdb/{name}")
    modal_app.runs_volume.commit()
    return result


@app.local_entrypoint()
def run(name: str = "sev-v03-devcal-20261004"):
    from decision_index.cyber.common import file_hash
    # Maximum two H100 hours; cpu + reserved maximum memory included.
    bound = 2 * (3.95 + 2 * .04730 + 128 * .008)
    print(f"One inference worker, maximum 7200 seconds, compute upper bound ${bound:.2f}; no training or test.")
    result = evaluate.remote(name, file_hash(INPUT))
    modal_app.pull_volume(f"/esdb/{name}", ROOT / "runs")
    print(result)
