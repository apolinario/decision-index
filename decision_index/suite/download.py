import gzip
import os
import shutil
from pathlib import Path

from decision_index import editions
from decision_index.suite.io import Suite

HUB = Path(__file__).resolve().parents[2] / "hub"


def hub_file(edition, name):
    e = editions.get(edition)
    path = HUB / name if e["id"] == "0.1" else HUB / e["id"] / name
    if not path.exists() and name == editions.EXCLUSIONS_FILE:
        path = HUB / name
    return path if path.exists() else None


def download(directory, dataset=None, revision=None, token=None, verify=True, edition=None):
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import hf_hub_download

    e = editions.get(edition)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in editions.files(e["id"]):
        target = directory / name
        if target.exists():
            continue
        local = hf_hub_download(dataset or e["dataset"], name, repo_type="dataset", revision=revision, token=token)
        shutil.copyfile(local, target)
    return Suite(directory, e["id"]).verify(strict=verify)


def copy_rows(src, dst):
    src = Path(src)
    if src.suffix == ".gz":
        shutil.copyfile(src, dst)
        return
    with src.open("rb") as f, gzip.open(dst, "wb", compresslevel=6) as g:
        shutil.copyfileobj(f, g)


def from_local(directory, rows, exclusions=None, manifest=None, verify=True, edition=None, added=None):
    e = editions.get(edition)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    copy_rows(rows, directory / editions.ROWS_FILE)
    if e["added_sha256"]:
        if not added:
            raise ValueError(f"edition {e['id']} needs --added-rows")
        copy_rows(added, directory / editions.ADDED_FILE)
    exclusions = exclusions or hub_file(e["id"], editions.EXCLUSIONS_FILE)
    manifest = manifest or hub_file(e["id"], editions.MANIFEST_FILE)
    if exclusions:
        shutil.copyfile(exclusions, directory / editions.EXCLUSIONS_FILE)
    if manifest:
        shutil.copyfile(manifest, directory / editions.MANIFEST_FILE)
    return Suite(directory, e["id"]).verify(strict=verify)
