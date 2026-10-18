"""Prepare uniformly rendered inference without changing selected Wyat weights."""

import argparse
import errno
import hashlib
import json
import os
import shutil
from pathlib import Path


def prepare(source, destination):
    from wyat.loading.fork_artifact import model_identity, validate_artifact

    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.is_relative_to(source):
        raise ValueError("Evaluation artifact must be outside the source artifact")
    manifest = validate_artifact(source)
    source_sha256 = hashlib.sha256(
        (source / "fork_manifest.json").read_bytes()
    ).hexdigest()
    # The official kit forbids per-benchmark prompts. Disable the release's
    # task-specific renderer uniformly, and give that inference artifact its
    # own native identity. Weight and tokenizer hashes remain unchanged.
    prepared = {
        **manifest,
        "input_rendering": None,
        "model_id": model_identity(manifest["files_sha256"], None),
        "provenance": {
            **manifest["provenance"],
            "decision_index_inference": {
                "source_model_id": manifest["model_id"],
                "source_manifest_sha256": source_sha256,
                "input_rendering": "verbatim-public-record-v1",
                "weights_and_tokenizer_unchanged": True,
            },
        },
    }
    destination.mkdir(parents=True, exist_ok=False)
    for name in manifest["files_sha256"]:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source / name, target)
        except OSError as error:
            if error.errno not in {errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP}:
                raise
            shutil.copyfile(source / name, target)
    (destination / "fork_manifest.json").write_text(
        json.dumps(prepared, indent=2) + "\n"
    )
    validate_artifact(destination)
    return {
        "source_model_id": manifest["model_id"],
        "evaluation_model_id": prepared["model_id"],
        "source_manifest_sha256": source_sha256,
        "evaluation_manifest_sha256": hashlib.sha256(
            (destination / "fork_manifest.json").read_bytes()
        ).hexdigest(),
        "input_rendering": "verbatim-public-record-v1",
        "weights_and_tokenizer_unchanged": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.out), indent=2))
