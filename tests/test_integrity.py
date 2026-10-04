import copy
import gzip
import hashlib
import json

import pytest

from decision_index import editions
from decision_index.runner import run
from decision_index.scoring.report import load_results, score
from decision_index.scoring.index import static_score, chance_baselines
from decision_index.suite.io import Suite


def fixture():
    return {"state": {}, "questions": {"q": {"type": "choice", "instructions": "Pick", "criteria": {"A": "a", "B": "b"}}},
            "expected": {"q": "A"}, "metadata": {},
            "_evaluation": {"run_id": "44:fixture", "catalog_id": 44, "dataset": "CLadder", "group_id": "fixture", "track": "all", "payload_sha256": "frozen"}}


def write_input(path):
    with gzip.open(path, "wt") as f:
        f.write(json.dumps(fixture()) + "\n")


def test_resume_rejects_settings_and_input_changes_without_relabeling(tmp_path):
    source, out = tmp_path / "rows.gz", tmp_path / "run"
    write_input(source)
    run("random", {"seed": 1}, source, out, warm=False, log=lambda _: None)
    old = {p.name: p.read_bytes() for p in out.iterdir()}
    with pytest.raises(ValueError, match="Resume"):
        run("random", {"seed": 5}, source, out, warm=False, log=lambda _: None)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == old
    with gzip.open(source, "wt") as f:
        row = fixture(); row["state"] = {"changed": True}; f.write(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Resume"):
        run("random", {"seed": 1}, source, out, warm=False, log=lambda _: None)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == old


def test_fresh_refuses_to_append_duplicate_predictions(tmp_path):
    source, out = tmp_path / "rows.gz", tmp_path / "run"
    write_input(source)
    run("random", {}, source, out, warm=False, log=lambda _: None)
    before = (out / "results.jsonl").read_bytes()
    with pytest.raises(FileExistsError, match="Fresh"):
        run("random", {}, source, out, resume=False, warm=False, log=lambda _: None)
    assert (out / "results.jsonl").read_bytes() == before


def test_resume_rejects_changed_engine_code_and_runtime(tmp_path, monkeypatch):
    from decision_index import runner
    from decision_index.engines.base import RandomEngine
    source, out = tmp_path / "rows.gz", tmp_path / "run"
    write_input(source)
    monkeypatch.setattr(RandomEngine, "runtime", lambda _: {"library": "version-one"})
    run("random", {}, source, out, warm=False, log=lambda _: None)
    old = {p.name: p.read_bytes() for p in out.iterdir()}
    original_hash = runner.sha256_file
    monkeypatch.setattr(runner, "sha256_file", lambda path: "changed-engine" if str(path).endswith("engines/base.py") else original_hash(path))
    with pytest.raises(ValueError, match="Resume"):
        run("random", {}, source, out, warm=False, log=lambda _: None)
    monkeypatch.setattr(runner, "sha256_file", original_hash)
    monkeypatch.setattr(RandomEngine, "runtime", lambda _: {"library": "version-two"})
    with pytest.raises(ValueError, match="runtime changed"):
        run("random", {}, source, out, warm=False, log=lambda _: None)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == old


def test_exclusion_hash_and_presence_are_required(tmp_path, monkeypatch):
    source = tmp_path / editions.ROWS_FILE
    write_input(source)
    exclusions = tmp_path / editions.EXCLUSIONS_FILE
    exclusions.write_text('{"rows": []}')
    pin = {**editions.EDITIONS["0.1"], "rows_gz_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
           "exclusions_sha256": hashlib.sha256(exclusions.read_bytes()).hexdigest()}
    monkeypatch.setitem(editions.EDITIONS, "0.1", pin)
    suite = Suite(tmp_path, "0.1")
    assert suite.verify()["match"]
    exclusions.write_text('{"rows": ["changed"]}')
    with pytest.raises(ValueError, match="hash mismatch"):
        suite.verify()
    exclusions.unlink()
    with pytest.raises(ValueError, match="hash mismatch"):
        suite.verify()
    with pytest.raises(FileNotFoundError, match="exclusions"):
        list(suite.rows(apply_exclusions=True))


def prediction():
    return {"run_id": "44:fixture", "catalog_id": 44, "payload_sha256": "frozen", "engine": "fixture", "status": "ok", "total_wall_ms": 1,
            "response": {"answers": {"q": {"type": "choice", "choice": "A", "probabilities": {"A": .9, "B": .9}}}}}


def test_imported_bad_distribution_is_failure_in_both_scorers():
    row, result = fixture(), prediction()
    report = score([row], {result["run_id"]: result})
    assert report["successful_requests"] == 0
    result["answers"] = result["response"]["answers"]
    report = static_score(44, [row], {result["run_id"]: result}, chance_baselines())
    assert report["coverage"] == report["raw"] == 0


def test_import_preserves_receipt_and_rejects_changed_payload_and_duplicate(tmp_path):
    result = prediction()
    source = tmp_path / "results.jsonl"
    source.write_text(json.dumps(result) + "\n")
    loaded = load_results(source)
    assert loaded[result["run_id"]]["payload_sha256"] == "frozen"
    row = fixture(); row["_evaluation"]["payload_sha256"] = "different"
    with pytest.raises(ValueError, match="receipt mismatch"):
        score([row], loaded)
    source.write_text((json.dumps(result) + "\n") * 2)
    with pytest.raises(ValueError, match="Repeated completed"):
        load_results(source)
    second = copy.deepcopy(result); second.update(run_id="44:other", engine="different")
    source.write_text(json.dumps(result) + "\n" + json.dumps(second) + "\n")
    with pytest.raises(ValueError, match="mix"):
        load_results(source)
