import os
import shutil
from pathlib import Path

from decision_index import constants as C
from decision_index.suite.io import Suite


def download(directory, dataset=C.SUITE_DATASET, revision=None, token=None, verify=True):
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import hf_hub_download

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in C.SUITE_FILES:
        target = directory / name
        if target.exists():
            continue
        local = hf_hub_download(dataset, name, repo_type="dataset", revision=revision, token=token)
        shutil.copyfile(local, target)
    suite = Suite(directory)
    return suite.verify(strict=verify)


def from_local(directory, rows, exclusions=None, manifest=None, verify=True):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(rows, directory / C.SUITE_ROWS_FILE)
    if exclusions:
        shutil.copyfile(exclusions, directory / C.SUITE_EXCLUSIONS_FILE)
    if manifest:
        shutil.copyfile(manifest, directory / C.SUITE_MANIFEST_FILE)
    return Suite(directory).verify(strict=verify)
