"""OpenRouter's System One-shaped Decisions API, not chat completions.

No automatic retries or fallback models. The append-only ledger includes failed
and ambiguous submissions, so resuming cannot silently spend a second time.
"""
import fcntl
import json
import math
import os
import time
from pathlib import Path

import httpx

from decision_index.engines.base import Engine, FatalEngineError, validate


MODELS = {
    "jev": "typesafe/jev-1.14",
    "mercury": "inception/mercury-decide",
}
ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
RESPONSE_MODELS = {
    "typesafe/jev-1.14": {"typesafe/jev-1.14", "typesafe/jev-1.14-20261014"},
    "inception/mercury-decide": {"inception/mercury-decide", "inception/mercury-decide-20260930"},
}


class OpenRouterDecisions(Engine):
    name = "openrouter"
    latency = "Client wall time including HTTP and validation; no retries or paid warmup."

    def __init__(self, model="jev", allow_spend=False, budget_usd=0, max_requests=100,
                 ledger=None, token_env="OPENROUTER_API_KEY", timeout=120, _transport=None):
        if allow_spend is not True:
            raise ValueError("OpenRouter requires --option allow_spend=true (including free routes)")
        if isinstance(budget_usd, bool) or not isinstance(budget_usd, (int, float)) or not math.isfinite(budget_usd) or budget_usd <= 0:
            raise ValueError("budget_usd must be positive and finite")
        if isinstance(max_requests, bool) or not isinstance(max_requests, int) or max_requests < 1:
            raise ValueError("max_requests must be a positive integer")
        if not ledger:
            raise ValueError("ledger is required; use a separate runs/<model>/requests.jsonl per configuration")
        if not isinstance(model, str) or model not in MODELS and model not in MODELS.values():
            raise ValueError("OpenRouter model must be jev, mercury, typesafe/jev-1.14, or inception/mercury-decide")
        self.model = MODELS.get(model, model)
        token = os.environ.get(token_env)
        if not token:
            raise ValueError(f"Set {token_env} in the environment; never pass the key as an engine option")
        self.provenance = {"kind": "openrouter-decisions", "endpoint": ENDPOINT, "model": self.model,
                           "protocol": "systemone", "adapter_version": 1, "fallbacks": False, "retries": 0}
        self.budget = budget_usd
        self.max_requests = max_requests
        self.path = Path(ledger).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.path.open("a+", encoding="utf-8")
        self.client = None
        try:
            fcntl.flock(self.log, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Persist the ledger's name and any newly created ancestor entries
            # before a paid submission can be reserved in this file.
            for directory in (self.path.parent, *self.path.parent.parents):
                fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            self.log.seek(0)
            events = [json.loads(line) for line in self.log]
            self.requests = 0
            self.spent = 0.0
            self.request_id = None
            self.recorded = {}
            submissions = {}
            pending = set()
            for event in events:
                if event["configuration"] != self.provenance:
                    raise ValueError("Ledger belongs to another configuration; use a new run directory")
                if event["event"] == "submitted":
                    self.requests += 1
                    pending.add(event["request"])
                    submissions[event["request"]] = event
                elif event["event"] == "response":
                    cost = event["cost_usd"]
                    if cost is not None:
                        self.spent += cost
                        pending.discard(event["request"])
                        submission = submissions[event["request"]]
                        if submission.get("run_id") is not None:
                            self.recorded[submission["run_id"]] = (submission["payload"], event)
            if pending:
                raise ValueError("Ledger contains an unresolved submission/cost. Reconcile it with OpenRouter before resuming; do not blindly retry.")
            self.client = httpx.Client(timeout=timeout, transport=_transport, headers={"Authorization": f"Bearer {token}"})
        except Exception:
            self.close()
            raise

    def _append(self, event):
        try:
            self.log.write(json.dumps({"configuration": self.provenance, **event}, ensure_ascii=False, allow_nan=False) + "\n")
            self.log.flush()
            os.fsync(self.log.fileno())
        except (OSError, ValueError, TypeError):
            raise FatalEngineError("Cannot durably write request ledger; stopping before any further submission") from None

    def set_request_id(self, run_id):
        self.request_id = run_id

    def verify_completed_request(self, run_id, state, questions):
        saved = self.recorded.get(run_id)
        payload = {"model": self.model, "state": state, "questions": questions,
                   "provider": {"allow_fallbacks": False}}
        if saved is None or saved[0] != payload:
            raise FatalEngineError("Completed run ID has no matching ledger payload; use a new run directory")

    def warmup(self):
        # Upstream calls warmup on every resume. It must not create paid requests.
        pass

    def __call__(self, state, questions):
        self.last_request_wall_ms = None
        self.response_replayed = False
        if not questions or any(q.get("type") not in ("choice", "noul") for q in questions.values()):
            raise ValueError("This public-suite engine accepts Choice and Noul only")
        payload = {"model": self.model, "state": state, "questions": questions,
                   "provider": {"allow_fallbacks": False}}
        # Check serialization before reserving a submission. No labels/metadata are sent.
        json.dumps(payload, allow_nan=False)
        if self.request_id in self.recorded:
            saved_payload, event = self.recorded[self.request_id]
            if saved_payload != payload:
                raise FatalEngineError("Run ID reused with different payload; use a new run directory")
            self.response_replayed = True
            self.last_request_wall_ms = event["http_wall_ms"]
            return self._decode(questions, event["raw"], event["http_status"], event["cost_usd"])
        if self.requests >= self.max_requests or self.spent >= self.budget:
            raise FatalEngineError("OpenRouter request limit or observed-cost budget reached")
        self.requests += 1
        request = self.requests
        self._append({"event": "submitted", "request": request, "run_id": self.request_id, "payload": payload})
        start = time.perf_counter()
        try:
            r = self.client.post(ENDPOINT, json=payload)
        except httpx.HTTPError:
            raise FatalEngineError("OpenRouter transport failure: submission may have been billed; inspect the unresolved ledger") from None
        try:
            raw = r.json()
        except ValueError:
            raw = {"unparsed_body": r.text}
        usage = raw.get("usage") if isinstance(raw, dict) else None
        cost = usage.get("cost") if isinstance(usage, dict) else None
        if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0:
            cost = None
        event = {"event": "response", "request": request, "http_status": r.status_code,
                 "http_wall_ms": (time.perf_counter() - start) * 1000, "cost_usd": cost, "raw": raw}
        self._append(event)
        if cost is not None:
            self.spent += cost
            if self.request_id is not None:
                self.recorded[self.request_id] = (payload, event)
        self.last_request_wall_ms = event["http_wall_ms"]
        return self._decode(questions, raw, r.status_code, cost)

    def _decode(self, questions, raw, status, cost):
        if not 200 <= status < 300 or not isinstance(raw, dict) or "error" in raw:
            raise FatalEngineError(f"OpenRouter rejected/failed request (HTTP {status}); inspect ledger; no automatic retry")
        if cost is None:
            raise FatalEngineError("OpenRouter response has no valid usage.cost; cost is unknown, stopping for reconciliation")
        if raw.get("model") not in RESPONSE_MODELS[self.model]:
            raise FatalEngineError("OpenRouter returned an unexpected model; inspect ledger; no automatic retry")
        response = {k: raw[k] for k in ("model", "answers", "usage", "provider", "id") if k in raw}
        validate(questions, response)
        # A chosen key must agree with its supplied distribution (ties are allowed).
        for key, q in questions.items():
            if q["type"] == "choice":
                answer = response["answers"][key]
                if answer["probabilities"][answer["choice"]] < max(answer["probabilities"].values()):
                    raise ValueError("Chosen answer is not a probability argmax")
        return response, raw

    def runtime(self):
        return {"request_ledger": str(self.path), "observed_cost_usd": self.spent,
                "submitted_requests": self.requests, "budget_usd": self.budget,
                "max_requests": self.max_requests}

    def close(self):
        if self.client is not None:
            self.client.close()
        self.log.close()
