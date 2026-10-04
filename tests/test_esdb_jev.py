import asyncio
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

spec = importlib.util.spec_from_file_location("esdb_jev", Path(__file__).parents[1] / "scripts/esdb_jev.py")
jev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jev)


def fixture():
    return {"state": {"events": ["e2", "e1"]}, "split": "calibration", "family": "endpoint_investigation",
            "questions": {"q": {"type": "choice", "instructions": "Pick", "criteria": {"B": "second", "A": "first"}}},
            "_evaluation": {"run_id": "fixture", "payload_sha256": "receipt"}}


def response(model=jev.MODEL, probabilities=None):
    return {"model": model, "answers": {"q": {"type": "choice", "choice": "B", "probabilities": probabilities or {"B": .8, "A": .2}}},
            "usage": {"input_tokens": 123}}


def call(handler):
    async def run():
        async with httpx.AsyncClient(base_url=jev.BASE_URL, transport=httpx.MockTransport(handler)) as client:
            return await jev.evaluate(client, fixture(), retries=0)
    return asyncio.run(run())


def test_request_sends_only_model_and_unmodified_ordered_evidence():
    def handler(request):
        body = json.loads(request.content)
        assert body == {"model": jev.MODEL, **jev.payload(fixture())}
        assert list(body["questions"]["q"]["criteria"]) == ["B", "A"]
        return httpx.Response(200, json=response(), headers={"x-typesafe-request-id": "provider-id"})
    result, fatal = call(handler)
    assert result["status"] == "ok" and not fatal
    assert result["input_tokens"] == 123
    assert result["request_sha256"] == jev.request_hash(fixture())
    assert result["attempts"][0]["request_id"] == "provider-id"


def test_foreign_model_stops_and_invalid_distribution_is_retained():
    result, fatal = call(lambda _: httpx.Response(200, json=response(model="unexpected")))
    assert result["status"] == "error" and fatal
    raw = response(probabilities={"B": .8, "A": .8})
    result, fatal = call(lambda _: httpx.Response(200, json=raw))
    assert result["status"] == "invalid" and not fatal
    assert result["raw_output"] == raw


def test_capacity_is_final_and_authorization_stops():
    result, fatal = call(lambda _: httpx.Response(422, text="maximum context length exceeded"))
    assert result["status"] == "unsupported" and not fatal
    result, fatal = call(lambda _: httpx.Response(401))
    assert result["status"] == "error" and fatal


def test_resume_rejects_duplicate_or_changed_receipts(tmp_path):
    result, _ = call(lambda _: httpx.Response(200, json=response()))
    result["run_identity_sha256"] = "identity"
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps(result) + "\n")
    assert len(jev.resume_results(path, [fixture()], "identity")) == 1
    with pytest.raises(ValueError, match="identity or input changed"):
        jev.resume_results(path, [fixture()], "changed")
    path.write_text((json.dumps(result) + "\n") * 2)
    with pytest.raises(ValueError, match="Repeated"):
        jev.resume_results(path, [fixture()], "identity")


def test_input_gate_rejects_test_or_leaked_gold(tmp_path):
    from decision_index.cyber.common import write_rows
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text('{}')
    source = tmp_path / "staged.jsonl.gz"
    row = {"id": "fixture", **fixture(), "expected": {"q": "B"}}
    write_rows(source, [row])
    write_rows(dataset / "inputs/development.jsonl.gz", [row])
    write_rows(dataset / "inputs/calibration.jsonl.gz", [])
    source.with_suffix(".receipt.json").write_text(json.dumps({"input_sha256": jev.file_hash(source),
        "dataset_manifest_sha256": jev.file_hash(dataset / "manifest.json"), "cases": 1, "gold_exposed": False}))
    with pytest.raises(ValueError, match="answer-free"):
        jev.load_inputs(source, dataset)
