import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from examples.prepare_wyat import prepare


def test_separate_manifest_preserves_selected_payload_and_source(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    data = b"unchanged selected weights"
    digest = hashlib.sha256(data).hexdigest()
    (source / "head.safetensors").write_bytes(data)
    manifest = {
        "model_id": "selected-identity",
        "input_rendering": {"mode": "api-bank-schema-next-v1"},
        "files_sha256": {"head.safetensors": digest},
        "provenance": {"checkpoint_sha256": "selected-checkpoint"},
    }
    original = json.dumps(manifest).encode()
    (source / "fork_manifest.json").write_bytes(original)
    validations = []

    def validate(path):
        validations.append(path)
        result = json.loads((path / "fork_manifest.json").read_text())
        assert (
            hashlib.sha256((path / "head.safetensors").read_bytes()).hexdigest()
            == digest
        )
        return result

    monkeypatch.setitem(
        sys.modules,
        "wyat.loading.fork_artifact",
        SimpleNamespace(
            validate_artifact=validate,
            model_identity=lambda files, policy: (
                "uniform-identity" if policy is None else None
            ),
        ),
    )
    destination = tmp_path / "uniform"
    receipt = prepare(source, destination)
    prepared = json.loads((destination / "fork_manifest.json").read_text())
    assert validations == [source, destination]
    assert (source / "fork_manifest.json").read_bytes() == original
    assert (source / "head.safetensors").read_bytes() == data
    assert prepared["files_sha256"] == manifest["files_sha256"]
    assert prepared["input_rendering"] is None
    assert prepared["model_id"] == "uniform-identity"
    assert prepared["provenance"]["checkpoint_sha256"] == "selected-checkpoint"
    assert receipt["source_model_id"] == "selected-identity"
    assert receipt["source_manifest_sha256"] == hashlib.sha256(original).hexdigest()
    with pytest.raises(FileExistsError):
        prepare(source, destination)
    with pytest.raises(ValueError, match="outside"):
        prepare(source, source / "nested")
