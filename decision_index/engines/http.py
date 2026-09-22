import os

from decision_index.engines.base import Engine, Unsupported

CAPACITY_MARKERS = (
    "options per choice",
    "a choice needs at least two options",
    "a score takes 2 to 10 levels",
    "the canvas holds",
    "maximum context length",
    "maximum model length",
    "longer than the maximum model length",
    "context window",
    "too many tokens",
)


class HttpSystemOne(Engine):
    name = "http"
    latency = "HTTP request wall time against the configured /v1/systemone endpoint, including server-side prompt construction and inference; excludes server startup."

    def __init__(self, base_url=None, model="default", token_env="DECISION_INDEX_API_KEY", timeout=600, extra=None, **options):
        super().__init__(**options)
        import httpx

        base_url = base_url or os.environ.get("DECISION_INDEX_BASE_URL")
        if not base_url:
            raise ValueError("HttpSystemOne needs base_url (or DECISION_INDEX_BASE_URL)")
        headers = {}
        token = os.environ.get(token_env)
        if token:
            headers["Authorization"] = "Bearer " + token
        self.model = model
        self.extra = extra or {}
        self.client = httpx.Client(base_url=base_url, timeout=timeout, headers=headers)
        self.provenance = {"kind": "http", "base_url": base_url, "model": model, "request_options": self.extra, "policy": "Unmodified state and questions sent as one /v1/systemone request; explicit capacity rejections (HTTP 422 with a known marker) are unsupported, other failures are errors."}

    def __call__(self, state, questions):
        r = self.client.post("/v1/systemone", json={"model": self.model, "state": state, "questions": questions, **self.extra})
        if r.status_code in (400, 413, 422):
            message = r.text
            if any(s in message for s in CAPACITY_MARKERS):
                raise Unsupported(message)
        r.raise_for_status()
        raw = r.json()
        return {k: v for k, v in raw.items() if k != "evaluation_trace"}, raw

    def close(self):
        self.client.close()
