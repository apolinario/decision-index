"""Self-hosted models in our evaluation matrix; no paid API routing."""
from urllib.parse import urlsplit

import httpx

from decision_index.engines.base import Unsupported
from decision_index.engines.http import HttpSystemOne


def loopback_url(url):
    parts = urlsplit(url)
    if parts.scheme != "http" or parts.hostname not in ("127.0.0.1", "localhost", "::1") or parts.username or parts.password or parts.query or parts.fragment or parts.path not in ("", "/"):
        raise ValueError("Local model URL must be an HTTP loopback origin without credentials or a path")
    return url


def check_server(engine):
    try:
        response = engine.client.get("/health")
        response.raise_for_status()
        health = response.json()
        if health.get("model") != engine.model or health.get("status") != "ok":
            raise ValueError(f"Expected a healthy {engine.model} server")
        engine.provenance["server"] = {k: v for k, v in health.items() if k not in ("stats", "queued")}
    except Exception:
        engine.close()
        raise


class _LocalDecision(HttpSystemOne):
    def __call__(self, state, questions):
        response, raw = super().__call__(state, questions)
        if response.get("model") != self.model:
            raise ValueError("Local server returned a different model")
        return response, raw


class LocalDecider(_LocalDecision):
    name = "pplx-decider"

    def __init__(self, base_url="http://127.0.0.1:8011", timeout=600):
        super().__init__(base_url=loopback_url(base_url), model="pplx-decider-v1.2",
                         endpoint="/v1/decisions", token_env="DECISION_INDEX_LOCAL_API_KEY", timeout=timeout)
        check_server(self)


class OllamaDecider(_LocalDecision):
    """Pinned v1.1 Q4_K_M, not the native v1.2 BF16 checkpoint."""
    name = "ollama-decider"
    digest = "733585a9ebb6205d2d5b64a13748f0b1c9f1285885dd267780cfdb1c4c8ef141"

    def __init__(self, base_url="http://127.0.0.1:11434", timeout=600, context=16384):
        if type(context) is not int or context not in (16384, 131072):
            raise ValueError("Ollama Decider context must be 16384 or 131072")
        model = "pplx-decider:27b-q4_K_M"
        if context == 131072:
            model += "-ctx128k"
            self.digest = "7002ddb908b18ed49c889f58b168af9e493d434ed1dfcc0426d03dc4f26c0ba6"
        super().__init__(base_url=loopback_url(base_url), model=model,
                         token_env="DECISION_INDEX_LOCAL_API_KEY", timeout=timeout)
        try:
            version = self.client.get("/api/version")
            version.raise_for_status()
            tags = self.client.get("/api/tags")
            tags.raise_for_status()
            model = next((m for m in tags.json()["models"] if m["name"] == self.model), None)
            if model is None or model.get("digest") != self.digest:
                raise ValueError("Expected the pinned Ollama Decider v1.1 Q4_K_M digest; pull the exact tag first")
            if model.get("details", {}).get("quantization_level") != "Q4_K_M":
                raise ValueError("Expected Q4_K_M quantization")
            self.provenance["server"] = {"version": version.json()["version"], "model": model}
            self.provenance["checkpoint_version"] = "v1.1"
            self.provenance["configured_context"] = context
        except Exception:
            self.close()
            raise

    def __call__(self, state, questions):
        if isinstance(questions, dict) and len(questions) > 64:
            raise Unsupported("Ollama supports at most 64 questions per request")
        try:
            return super().__call__(state, questions)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 400:
                try:
                    error = exc.response.json().get("error")
                except (ValueError, AttributeError):
                    error = None
                if error == "state must not be empty":
                    raise Unsupported("Ollama does not support the suite's empty-state requests") from exc
            raise

class LocalGlide(_LocalDecision):
    name = "glide"

    def __init__(self, base_url="http://127.0.0.1:8012", timeout=600):
        super().__init__(base_url=loopback_url(base_url), model="glide-v1.2",
                         token_env="DECISION_INDEX_LOCAL_API_KEY", timeout=timeout)
        check_server(self)


class LocalStartLux(_LocalDecision):
    name = "startlux"

    def __init__(self, base_url="http://127.0.0.1:8090", timeout=600):
        super().__init__(base_url=loopback_url(base_url),
                         model="startlux-decision-v1.1-27b-instant-q4-k-m",
                         token_env="DECISION_INDEX_LOCAL_API_KEY", timeout=timeout)
        check_server(self)
