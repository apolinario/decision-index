"""HTTP server: ``POST /v1/systemone`` and ``GET /v1/models``.

Checks run in the order a client can act on them: 401 (auth, when keys are
configured), 429 (per-key rate limit), 422 (validation), 503 (inference queue
full). Requests reach vLLM together so it can batch them; deterministic
requests run alone, since batching makes answers depend on traffic.
"""

from __future__ import annotations

import hmac
import json
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from datetime import date
from typing import Any

import anyio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .engine import DecisionEngine, InvalidRequest
from .schema import (
    DecisionRequest,
    ModelMetadata,
    ModelMetadataList,
    error_body,
)

RATE_WINDOW_SECONDS = 60.0


class RateLimiter:
    """Sliding one-minute window of request times per caller."""

    def __init__(self, per_minute: int, clock: Callable[[], float] = time.monotonic):
        if per_minute < 1:
            raise ValueError(f"per_minute must be at least 1, got {per_minute}")
        self.per_minute = per_minute
        self.clock = clock
        self._seen: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, caller: str) -> bool:
        now = self.clock()
        with self._lock:
            times = self._seen.setdefault(caller, deque())
            while times and now - times[0] >= RATE_WINDOW_SECONDS:
                times.popleft()
            if len(times) >= self.per_minute:
                return False
            times.append(now)
            return True


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"detail": message}, status_code=status)


class InferenceGate:
    """Shared entry for requests the backend may overlap, exclusive entry for
    the rest. A waiting exclusive request holds back new shared ones, so
    steady shared traffic cannot starve it."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._shared = 0
        self._exclusive = False
        self._waiting = 0

    @contextmanager
    def shared(self) -> Iterator[None]:
        with self._cond:
            self._cond.wait_for(lambda: not self._exclusive and not self._waiting)
            self._shared += 1
        try:
            yield
        finally:
            with self._cond:
                self._shared -= 1
                self._cond.notify_all()

    @contextmanager
    def exclusive(self) -> Iterator[None]:
        with self._cond:
            self._waiting += 1
            try:
                self._cond.wait_for(lambda: not self._exclusive and not self._shared)
            finally:
                self._waiting -= 1
            self._exclusive = True
        try:
            yield
        finally:
            with self._cond:
                self._exclusive = False
                self._cond.notify_all()


def _bearer(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    return token.strip() if scheme.lower() == "bearer" and token.strip() else None


def create_app(
    engine: DecisionEngine,
    *,
    release_date: date,
    api_keys: Iterable[str] = (),
    rate_limit_per_minute: int | None = None,
    max_pending: int = 8,
    clock: Callable[[], float] = time.monotonic,
) -> FastAPI:
    """Build the app around a ready engine. With no ``api_keys`` the server is
    open; with no ``rate_limit_per_minute`` it is unlimited. ``release_date`` is
    what ``GET /v1/models`` reports for the served model."""
    if max_pending < 1:
        raise ValueError(f"max_pending must be at least 1, got {max_pending}")
    keys = [k.encode() for k in api_keys]
    limiter = (
        RateLimiter(rate_limit_per_minute, clock) if rate_limit_per_minute else None
    )
    gate = InferenceGate()
    pending = threading.BoundedSemaphore(max_pending)
    # One worker thread per admitted request; anyio's shared pool has 40.
    workers: list[anyio.CapacityLimiter] = []
    app = FastAPI(
        title="Nexus SmallDecision System One server", docs_url=None, redoc_url=None
    )
    listing = ModelMetadataList(
        models=[
            ModelMetadata(
                name=engine.model_id,
                description=(
                    "Nexus SmallDecision System One model: calibrated noul, "
                    "choice and score answers read from one answer token per "
                    "question."
                ),
                release_date=release_date.isoformat(),
            )
        ]
    ).model_dump()

    def caller(request: Request) -> str | JSONResponse:
        """The caller's identity, or the 401 response for a bad key."""
        token = _bearer(request)
        if keys:
            if token is None or not any(
                hmac.compare_digest(token.encode(), k) for k in keys
            ):
                return _error(401, "missing or invalid API key")
            return token
        return request.client.host if request.client else "anonymous"

    def run(parsed: DecisionRequest) -> dict[str, Any]:
        deterministic = parsed.x_nexus is not None and parsed.x_nexus.deterministic
        with gate.exclusive() if deterministic else gate.shared():
            response = engine.decide(parsed)
        exclude = {"x_nexus"} if response.x_nexus is None else set()
        return response.model_dump(mode="json", exclude=exclude)

    @app.get("/v1/models")
    def list_models(request: Request) -> Any:
        who = caller(request)
        if isinstance(who, JSONResponse):
            return who
        return listing

    @app.post("/v1/systemone")
    async def systemone(request: Request) -> Any:
        who = caller(request)
        if isinstance(who, JSONResponse):
            return who
        if limiter is not None and not limiter.allow(who):
            return _error(429, "rate limit exceeded")
        try:
            payload = json.loads(await request.body())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            return JSONResponse(
                error_body(
                    [
                        {
                            "loc": (),
                            "msg": f"body is not valid JSON: {exc}",
                            "type": "json_invalid",
                        }
                    ]
                ),
                status_code=422,
            )
        try:
            parsed = DecisionRequest.model_validate(payload)
        except ValidationError as exc:
            return JSONResponse(error_body(exc.errors()), status_code=422)
        if not pending.acquire(blocking=False):
            return _error(503, "inference queue is full")
        if not workers:
            workers.append(anyio.CapacityLimiter(max_pending))
        try:
            body = await anyio.to_thread.run_sync(run, parsed, limiter=workers[0])
        except InvalidRequest as exc:
            return JSONResponse(
                error_body(
                    [
                        {
                            "loc": exc.loc,
                            "msg": str(exc),
                            "type": "invalid_request",
                        }
                    ]
                ),
                status_code=422,
            )
        finally:
            pending.release()
        return JSONResponse(body)

    return app
