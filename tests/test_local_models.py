import json
import threading
from contextlib import contextmanager

import httpx
import pytest

from decision_index.engines import Unsupported, load_engine
from decision_index.engines.local_http import loopback_url
from decision_index.pplx_server import MODEL, DeciderService, create_server, validate_request


BODY = {"model": MODEL, "state": {"ticket": "duplicate charge"}, "questions": {
    "team": {"type": "choice", "criteria": {"billing": "Money", "support": "Bugs"}},
    "urgent": {"type": "noul", "instructions": "Is the customer blocked?"},
    "severity": {"type": "score", "criteria": ["Low", "Medium", "High"]}}}


class FakeDecider:
    def __init__(self):
        self.calls = []

    def decide(self, request):
        self.calls.append(request)
        return {"answers": {
            "team": {"type": "choice", "choice": "billing", "confidence": 0.999,
                     "probabilities": {"billing": 0.8, "support": 0.2}},
            "urgent": {"type": "noul", "noul": 0.75},
            "severity": {"type": "score", "score": 0, "confidence": 0,
                         "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}}},
                "usage": {"input_tokens": 231, "output_tokens": 0, "sequences": 1}}


@contextmanager
def running_server(service):
    server = create_server(service, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_native_joint_call_and_perplexity_summaries():
    decider = FakeDecider()
    service = DeciderService(decider)
    response = service.decide(BODY)
    assert decider.calls == [{"state": BODY["state"], "questions": BODY["questions"]}]
    assert response["model"] == MODEL
    assert response["answers"]["team"]["confidence"] == pytest.approx(0.6)
    assert response["answers"]["severity"]["score"] == pytest.approx(1.6)
    assert response["answers"]["severity"]["confidence"] == pytest.approx(0.6)
    assert response["answers"]["severity"]["probabilities"] == {"0": 0.1, "1": 0.2, "2": 0.7}
    assert response["usage"] == {"input_tokens": 231, "output_tokens": 3}
    assert "cost" not in response["usage"]


@pytest.mark.parametrize("body", [
    {**BODY, "state": None}, {**BODY, "questions": {}}, {**BODY, "model": "pplx-decider-latest"},
    {**BODY, "temperature": 0}, {**BODY, "images": []},
    {**BODY, "state": {"photo": {"type": "image_url", "image_url": {"url": "https://example.org/x.png"}}}},
    {**BODY, "questions": {"x": {"type": "noul"}}},
    {**BODY, "questions": {"x": {"type": "score", "criteria": ["Only"]}}},
    {**BODY, "questions": {"": {"type": "noul", "instructions": "yes?"}}},
    {**BODY, "questions": {"x": {"type": "choice", "criteria": {}}}},
    {**BODY, "questions": {"x": {"type": "score", "criteria": [None, "High"]}}},
])
def test_local_api_rejects_unsupported_or_invalid_requests(body):
    with pytest.raises(ValueError):
        validate_request(body)


def test_http_contract_without_gpu():
    decider = FakeDecider()
    with running_server(DeciderService(decider)) as url, httpx.Client(base_url=url) as client:
        r = client.post("/v1/decisions", json=BODY)
        assert r.status_code == 200 and r.headers["x-request-id"]
        assert r.json()["usage"]["output_tokens"] == 3
        assert client.get("/health").json()["model"] == MODEL
        assert client.post("/v1/decisions/", json=BODY).status_code == 404
        assert client.post("/v1/systemone", json=BODY).status_code == 404
        r = client.post("/v1/decisions", json={**BODY, "unknown": True})
        assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request"
        assert len(decider.calls) == 1
        r = client.post("/v1/decisions", content='{"state":NaN}')
        assert r.status_code == 400


@pytest.mark.parametrize("url", ["https://api.perplexity.ai", "http://example.com", "http://localhost:8011/path",
                                 "http://user:password@localhost:8011", "http://localhost?key=secret"])
def test_local_profiles_cannot_route_to_paid_providers(url):
    with pytest.raises(ValueError):
        loopback_url(url)


@pytest.mark.parametrize("name,model,endpoint", [("pplx-decider", MODEL, "/v1/decisions"),
                                                 ("glide", "glide-v1.2", "/v1/systemone"),
                                                 ("startlux", "startlux-decision-v1.1-27b-instant-q4-k-m", "/v1/systemone")])
def test_local_engine_profiles(monkeypatch, name, model, endpoint):
    monkeypatch.setenv("OPENROUTER_API_KEY", "never-send-this")
    monkeypatch.setenv("DECISION_INDEX_API_KEY", "also-do-not-send")
    monkeypatch.delenv("DECISION_INDEX_LOCAL_API_KEY", raising=False)
    calls = []

    def handler(request):
        assert "authorization" not in request.headers
        calls.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"status": "ok", "model": model})
        assert request.url.path == endpoint
        assert json.loads(request.content)["model"] == model
        return httpx.Response(200, json={"model": model, "answers": {}, "usage": {"input_tokens": 1}})

    client_type = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: client_type(**kw, transport=httpx.MockTransport(handler)))
    e = load_engine(name)
    try:
        e("context", {})
        assert e.provenance["endpoint"] == endpoint
        assert e.provenance["server"]["model"] == model
    finally:
        e.close()
    assert len(calls) == 2


def test_bad_server_identity_prevents_evaluation(monkeypatch):
    client_type = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: client_type(**kw, transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json={"status": "ok", "model": "wrong"}))))
    with pytest.raises(ValueError, match="healthy"):
        load_engine("pplx-decider")


@pytest.mark.parametrize("valid", [True, False])
@pytest.mark.parametrize("context", [16384, 131072])
def test_ollama_decider_pins_local_model(monkeypatch, valid, context):
    from decision_index.engines.local_http import OllamaDecider
    calls = []
    model = "pplx-decider:27b-q4_K_M" + ("-ctx128k" if context == 131072 else "")
    digest = OllamaDecider.digest if context == 16384 else "7002ddb908b18ed49c889f58b168af9e493d434ed1dfcc0426d03dc4f26c0ba6"
    monkeypatch.setenv("OPENROUTER_API_KEY", "never-send")
    monkeypatch.setenv("DECISION_INDEX_API_KEY", "never-send")
    monkeypatch.delenv("DECISION_INDEX_LOCAL_API_KEY", raising=False)

    def handler(request):
        assert "authorization" not in request.headers
        calls.append(request.url.path)
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.40.12"})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": model,
                "digest": digest if valid else "wrong",
                "details": {"quantization_level": "Q4_K_M"}}]})
        assert request.url.path == "/v1/systemone"
        assert json.loads(request.content)["model"] == model
        if json.loads(request.content)["state"] == "over-limit":
            return httpx.Response(413, json={"error": "text and schema must not exceed 64 KiB"})
        if json.loads(request.content)["state"] == "":
            return httpx.Response(400, json={"error": "state must not be empty"})
        if json.loads(request.content)["state"] == "malformed":
            return httpx.Response(400, json={"error": "invalid question"})
        return httpx.Response(200, json={"model": model, "answers": {}})

    client_type = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: client_type(**kw, transport=httpx.MockTransport(handler)))
    if not valid:
        with pytest.raises(ValueError, match="digest"):
            load_engine("ollama-decider", context=context)
        assert calls == ["/api/version", "/api/tags"]
        return
    e = load_engine("ollama-decider", context=context)
    try:
        e("state", {})
        assert e.provenance["checkpoint_version"] == "v1.1"
        assert e.provenance["server"]["model"]["digest"] == digest
        assert e.provenance["configured_context"] == context
        with pytest.raises(Unsupported, match="64 questions"):
            e("state", {str(i): {} for i in range(65)})
        with pytest.raises(Unsupported, match="64 KiB"):
            e("over-limit", {})
        with pytest.raises(Unsupported, match="empty-state"):
            e("", {})
        with pytest.raises(httpx.HTTPStatusError):
            e("malformed", {})
    finally:
        e.close()


@pytest.mark.parametrize("context", [True, 32768, 131072.0, "131072"])
def test_ollama_decider_rejects_unpinned_context(context):
    with pytest.raises(ValueError, match="context"):
        load_engine("ollama-decider", context=context)


@pytest.mark.parametrize("questions", [
    {str(i): {"type": "noul", "instructions": "True?"} for i in range(129)},
    {"x": {"type": "choice", "criteria": {str(i): str(i) for i in range(256)}}},
    {"x": {"type": "score", "criteria": ["Only"]}},
    {"x": {"type": "score", "criteria": [str(i) for i in range(11)]}},
])
def test_capacity_limits_are_unsupported_end_to_end(questions):
    decider = FakeDecider()
    with running_server(DeciderService(decider)) as url:
        e = load_engine("pplx-decider", base_url=url)
        try:
            with pytest.raises(Unsupported):
                e("context", questions)
        finally:
            e.close()
    assert not decider.calls


@pytest.mark.parametrize("questions", [{}, {"x": {"type": "choice", "criteria": {}}},
                                       {"x": {"type": "score", "criteria": []}},
                                       {"x": {"type": "noul"}}])
def test_malformed_requests_stay_errors_end_to_end(questions):
    decider = FakeDecider()
    with running_server(DeciderService(decider)) as url:
        e = load_engine("pplx-decider", base_url=url)
        try:
            with pytest.raises(httpx.HTTPStatusError):
                e("context", questions)
        finally:
            e.close()
    assert not decider.calls
