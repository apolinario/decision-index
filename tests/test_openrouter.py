import json

import httpx
import pytest

from decision_index.engines import FatalEngineError, load_engine
from decision_index.engines.openrouter import ENDPOINT, OpenRouterDecisions
from decision_index.runner import run


QUESTIONS = {
    "route": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Charges", "support": "Bugs"}},
    "urgent": {"type": "noul", "instructions": "Is the customer blocked?"},
}


def result(cost=0.01):
    return {"id": "gen-test", "model": "typesafe/jev-1.14-20261014", "provider": "TypeSafe",
            "answers": {"route": {"type": "choice", "choice": "billing", "confidence": 0.6,
                                  "probabilities": {"billing": 0.8, "support": 0.2}},
                        "urgent": {"type": "noul", "noul": 0.9}},
            "usage": {"input_tokens": 100, "output_tokens": 0, "cost": cost}}


@pytest.fixture
def options(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    return {"allow_spend": True, "budget_usd": 0.02, "max_requests": 2,
            "ledger": str(tmp_path / "requests.jsonl")}


def engine(options, handler=None):
    return OpenRouterDecisions(**options, _transport=httpx.MockTransport(handler or (lambda _: httpx.Response(200, json=result()))))


def test_request_and_native_response(options):
    requests = []

    def handler(request):
        requests.append(request)
        assert str(request.url) == ENDPOINT
        assert request.headers["authorization"] == "Bearer test-secret"
        payload = json.loads(request.content)
        assert payload == {"model": "typesafe/jev-1.14", "state": {"ticket": "double charge"},
                           "questions": QUESTIONS, "provider": {"allow_fallbacks": False}}
        return httpx.Response(200, json=result())

    e = engine(options, handler)
    try:
        e.warmup()
        assert not requests
        response, raw = e({"ticket": "double charge"}, QUESTIONS)
        assert response["provider"] == "TypeSafe"
        assert raw == result()
        assert response["answers"]["route"]["confidence"] == 0.6
        assert e.runtime()["observed_cost_usd"] == 0.01
    finally:
        e.close()
    log = open(options["ledger"]).read()
    assert "test-secret" not in log
    assert '"event": "submitted"' in log and '"event": "response"' in log


@pytest.mark.parametrize("override", [{"allow_spend": False}, {"budget_usd": 0}, {"budget_usd": float("nan")},
                                      {"max_requests": 0}, {"max_requests": True}, {"ledger": None},
                                      {"model": "~typesafe/jev-latest"}, {"model": "openai/gpt-6-luna-decisions"},
                                      {"model": "liquid/d1-ultra"}, {"model": "perplexity/pplx-decider-v1.2"},
                                      {"model": "baseten/mammoth-1:free"}, {"model": "openai/gpt-4o"}])
def test_preflight_rejects_unsafe_options(options, override):
    with pytest.raises(ValueError):
        engine({**options, **override})


def test_key_required(options, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(ValueError, match="Set OPENROUTER_API_KEY"):
        engine(options)


def test_persistent_budget_and_configuration(options):
    e = engine(options)
    e("ticket", QUESTIONS)
    e.close()
    e = engine(options)
    assert e.requests == 1
    e("ticket", QUESTIONS)
    with pytest.raises(FatalEngineError, match="limit"):
        e("ticket", QUESTIONS)
    e.close()
    with pytest.raises(ValueError, match="another configuration"):
        engine({**options, "model": "mercury"})


def test_ledger_lock(options):
    e = engine(options)
    try:
        with pytest.raises(BlockingIOError):
            engine(options)
    finally:
        e.close()


@pytest.mark.parametrize("mode", ["timeout", "missing-cost", "error", "non-json"])
def test_ambiguous_effects_stop_and_block_resume(options, mode):
    calls = []

    def handler(request):
        calls.append(request)
        if mode == "timeout":
            raise httpx.ReadTimeout("do not log credentials", request=request)
        if mode == "error":
            return httpx.Response(429, json={"error": {"message": "rate limited"}})
        if mode == "non-json":
            return httpx.Response(502, text="gateway unavailable")
        return httpx.Response(200, json=result(None))

    e = engine(options, handler)
    with pytest.raises(FatalEngineError):
        e("ticket", QUESTIONS)
    e.close()
    assert len(calls) == 1
    with pytest.raises(ValueError, match="unresolved"):
        engine(options, handler)
    assert len(calls) == 1


@pytest.mark.parametrize("mode", ["missing-option", "bad-choice", "nan", "missing-answer"])
def test_invalid_probabilities_are_not_repaired(options, mode):
    data = result()
    if mode == "missing-option":
        del data["answers"]["route"]["probabilities"]["support"]
    elif mode == "bad-choice":
        data["answers"]["route"]["choice"] = "support"
    elif mode == "nan":
        data["answers"]["urgent"]["noul"] = 2
    else:
        del data["answers"]["urgent"]
    e = engine(options, lambda _: httpx.Response(200, json=data))
    try:
        with pytest.raises(ValueError):
            e("ticket", QUESTIONS)
        assert e.spent == 0.01  # invalid answers still cost money
    finally:
        e.close()


def test_zero_cost_and_registry(options):
    data = {**result(0), "model": "inception/mercury-decide-20260930"}
    e = load_engine("openrouter", **options, model="mercury",
                     _transport=httpx.MockTransport(lambda _: httpx.Response(200, json=data)))
    try:
        e("ticket", QUESTIONS)
        assert e.spent == 0
    finally:
        e.close()


def test_crash_recovery_replays_by_run_id_before_budget_check(options):
    e = engine({**options, "max_requests": 1})
    e.set_request_id("benchmark:case:1")
    original, _ = e("ticket", QUESTIONS)
    original_timing = e.last_request_wall_ms
    e.close()  # simulate crash before upstream writes results.jsonl

    def must_not_submit(request):
        pytest.fail("Recovery must not submit a second billed request")

    e = engine({**options, "max_requests": 1}, must_not_submit)
    try:
        e.set_request_id("benchmark:case:1")
        recovered, _ = e("ticket", QUESTIONS)
        assert recovered == original
        assert e.response_replayed and e.last_request_wall_ms == original_timing
        assert e.requests == 1 and e.spent == 0.01
        with pytest.raises(FatalEngineError, match="different payload"):
            e("changed ticket", QUESTIONS)
        e.set_request_id("benchmark:case:2")
        with pytest.raises(FatalEngineError, match="limit"):
            e("ticket", QUESTIONS)
    finally:
        e.close()


def test_ledger_persistence_failure_is_fatal(options, monkeypatch):
    e = engine(options)
    calls = []
    monkeypatch.setattr("decision_index.engines.openrouter.os.fsync", lambda _: (_ for _ in ()).throw(OSError("disk full")))
    e.client.close()
    e.client = httpx.Client(transport=httpx.MockTransport(lambda r: calls.append(r)))
    try:
        with pytest.raises(FatalEngineError, match="durably"):
            e("ticket", QUESTIONS)
        assert not calls
    finally:
        e.close()


def test_runner_records_fatal_and_stops(tmp_path, monkeypatch):
    class Stopped:
        provenance = {}
        latency = "test"
        closed = False
        calls = 0

        def synchronize(self): pass
        def warmup(self): pass
        def runtime(self): return {}
        def close(self): self.closed = True
        def __call__(self, **payload):
            self.calls += 1
            raise FatalEngineError("budget reached")

    e = Stopped()
    monkeypatch.setattr("decision_index.runner.load_engine", lambda *a, **kw: e)
    rows = tmp_path / "rows.jsonl"
    rows.write_text("\n".join(json.dumps({"state": "ticket", "questions": QUESTIONS,
                                          "_evaluation": {"run_id": str(i)}}) for i in range(3)))
    with pytest.raises(FatalEngineError):
        run("openrouter", {}, rows, tmp_path / "run", log=lambda _: None)
    assert e.calls == 1 and e.closed
    saved = [json.loads(line) for line in (tmp_path / "run/results.jsonl").read_text().splitlines()]
    assert len(saved) == 1 and saved[0]["status"] == "error"


@pytest.mark.parametrize("model", [None, "inception/mercury-decide", "typesafe/jev-1.14-other"])
def test_returned_model_must_match_requested_profile(options, model):
    e = engine(options, lambda _: httpx.Response(200, json={**result(), "model": model}))
    try:
        with pytest.raises(FatalEngineError, match="unexpected model"):
            e("ticket", QUESTIONS)
        assert e.spent == 0.01
    finally:
        e.close()


def test_directory_sync_failure_prevents_client_creation(options, monkeypatch):
    monkeypatch.setattr("decision_index.engines.openrouter.os.fsync", lambda _: (_ for _ in ()).throw(OSError("disk failure")))
    with pytest.raises(OSError):
        engine(options)


def test_runner_checks_completed_payload_before_skipping(options, tmp_path, monkeypatch):
    calls = []
    def load(*args, **kwargs):
        return engine(options, lambda request: (calls.append(request) or httpx.Response(200, json=result())))
    monkeypatch.setattr("decision_index.runner.load_engine", load)
    rows = tmp_path / "rows.jsonl"
    row = {"state": "ticket", "questions": QUESTIONS, "_evaluation": {"run_id": "one"}}
    rows.write_text(json.dumps(row) + "\n")
    run("openrouter", {}, rows, tmp_path / "run", compact=True, log=lambda _: None)
    run("openrouter", {}, rows, tmp_path / "run", compact=True, log=lambda _: None)
    rows.write_text(json.dumps({**row, "state": "changed"}) + "\n")
    with pytest.raises(FatalEngineError, match="matching ledger payload"):
        run("openrouter", {}, rows, tmp_path / "run", compact=True, log=lambda _: None)
    assert len(calls) == 1
