import os
import shlex
import subprocess
import tarfile
import tempfile
from pathlib import Path

DEFAULT_FLAVOR = "rtx-pro-6000"
DEFAULT_IMAGE = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime"
DEFAULT_TIMEOUT = "72h"


def package_source(root):
    root = Path(root)
    tmp = Path(tempfile.mkdtemp()) / "decision-index-src.tar.gz"
    with tarfile.open(tmp, "w:gz") as tar:
        for name in ("pyproject.toml", "README.md", "LICENSE", "decision_index"):
            path = root / name
            if path.exists():
                tar.add(path, arcname="decision-index/" + name, filter=lambda ti: None if "__pycache__" in ti.name else ti)
    return tmp


def job_command(results_repo, engine, engine_args, suite_dataset, run_name, extra_run_args, git_url=None):
    install = f"pip install -q 'git+{git_url}#egg=decision-index[transformers]'" if git_url else "hf download " + shlex.quote(results_repo) + " code/decision-index-src.tar.gz --repo-type dataset --local-dir /tmp/src && tar xzf /tmp/src/code/decision-index-src.tar.gz -C /tmp/src && pip install -q '/tmp/src/decision-index[transformers]'"
    run = " ".join(["python -m decision_index pipeline", "--engine", shlex.quote(engine), *[shlex.quote(a) for a in engine_args], "--suite-dataset", shlex.quote(suite_dataset), "--out", shlex.quote("/tmp/runs/" + run_name), "--upload", shlex.quote(results_repo), "--upload-path", shlex.quote("runs/" + run_name), *[shlex.quote(a) for a in extra_run_args]])
    return "set -euo pipefail; export HF_HUB_DISABLE_XET=1; pip install -q -U huggingface_hub && " + install + " && " + run


def submit(results_repo, engine, engine_args, suite_dataset, run_name, extra_run_args=(), flavor=DEFAULT_FLAVOR, image=DEFAULT_IMAGE, timeout=DEFAULT_TIMEOUT, git_url=None, namespace=None, private=True, token=None, dry_run=False, source_root=None):
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import HfApi, get_token

    token = token or os.environ.get("HF_TOKEN") or get_token()
    api = HfApi(token=token)
    command = job_command(results_repo, engine, list(engine_args), suite_dataset, run_name, list(extra_run_args), git_url)
    if dry_run:
        return {"flavor": flavor, "image": image, "timeout": timeout, "command": command, "results_repo": results_repo}
    api.create_repo(results_repo, repo_type="dataset", private=private, exist_ok=True)
    if not git_url:
        root = Path(source_root) if source_root else Path(__file__).resolve().parents[1]
        tarball = package_source(root)
        api.upload_file(path_or_fileobj=str(tarball), path_in_repo="code/decision-index-src.tar.gz", repo_id=results_repo, repo_type="dataset", commit_message="decision-index source snapshot")
    job = api.run_job(image=image, command=["bash", "-lc", command], flavor=flavor, timeout=timeout, secrets={"HF_TOKEN": token}, env={"HF_HUB_DISABLE_XET": "1", "PYTHONUNBUFFERED": "1"}, namespace=namespace, labels={"decision-index": run_name})
    return {"job_id": getattr(job, "id", None), "url": getattr(job, "url", None), "flavor": flavor, "image": image, "timeout": timeout, "results_repo": results_repo, "command": command}


def logs(job_id, namespace=None, token=None):
    return subprocess.run(["hf", "jobs", "logs", job_id] + (["--namespace", namespace] if namespace else []), check=False)
