"""Answer-slot logits from a running vLLM OpenAI-compatible server, one
completion prompt per branch.

The server must run with ``--logprobs-mode raw_logits``, so ``logprob_token_ids``
returns the raw logits of exactly the answer tokens. This client only connects:
it never starts or stops a server. Text state only.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import httpx

from .engine import ContextTooLong, TokenizerView
from .render import RenderedRequest

# vLLM's per-request cap on logprob_token_ids (sampling_params.py,
# MAX_LOGPROB_TOKEN_IDS); wider answer sets are split across requests.
MAX_TOKEN_IDS = 128
_TOKEN_KEY = re.compile(r"token_id:(\d+)")


def default_client() -> httpx.Client:
    """No connection cap: concurrent branch requests must not queue behind a
    client-side pool limit."""
    return httpx.Client(
        timeout=600,
        limits=httpx.Limits(max_connections=None, max_keepalive_connections=256),
    )


def parse_logits(choice: dict, token_ids: Sequence[int]) -> dict[int, float]:
    """Raw logits of ``token_ids`` from one completion choice."""
    logprobs = choice.get("logprobs") or {}
    top = logprobs.get("top_logprobs") or []
    if not top:
        raise ValueError("completion carries no top_logprobs")
    found: dict[int, float] = {}
    for key, value in top[0].items():
        m = _TOKEN_KEY.fullmatch(key)
        if m is None:
            raise ValueError(f"unexpected logprob key {key!r}; need token-id keys")
        found[int(m.group(1))] = float(value)
    missing = [t for t in token_ids if t not in found]
    if missing:
        raise ValueError(f"vLLM returned no logit for token ids {missing[:10]}")
    return {t: found[t] for t in token_ids}


class VllmBackend:
    name = "vllm"

    def __init__(
        self,
        view: TokenizerView,
        *,
        base_url: str,
        served_model: str,
        max_context: int,
        client: httpx.Client | None = None,
        concurrency: int = 16,
    ):
        self.view = view
        self.base_url = base_url.rstrip("/")
        self.served_model = served_model
        self.max_context = max_context
        self.client = client or default_client()
        self.concurrency = concurrency

    def token_id(self, token: str) -> int:
        ids = self.view.encode(token)
        if len(ids) != 1:
            raise ValueError(f"answer form {token!r} is not a single token")
        return ids[0]

    def _post(self, body: dict) -> dict:
        r = self.client.post(
            f"{self.base_url}/v1/completions",
            json={
                "model": self.served_model,
                "max_tokens": 1,
                "temperature": 0.0,
                **body,
            },
        )
        if r.status_code != 200:
            raise RuntimeError(
                f"vLLM completion failed ({r.status_code}): {r.text[:500]}"
            )
        return r.json()

    def _complete(self, prompt: list[int], token_ids: list[int]) -> dict[int, float]:
        body = self._post(
            {
                "prompt": prompt,
                "logprobs": 1,
                "logprob_token_ids": token_ids,
                "return_tokens_as_token_ids": True,
            }
        )
        (choice,) = body["choices"]
        return parse_logits(choice, token_ids)

    def branch_logits(
        self, rendered: RenderedRequest, *, deterministic: bool = False
    ) -> tuple[list[list[float]], int]:
        """Answer logits per branch and the total input token count. In
        deterministic mode branches go one at a time; otherwise concurrently,
        so the server batches them and reuses the cached prefix."""
        prefix_ids = self.view.encode(rendered.prefix)
        branch_ids = [self.view.encode(b.text) for b in rendered.branches]
        longest = max(len(ids) for ids in branch_ids)
        if len(prefix_ids) + longest > self.max_context:
            raise ContextTooLong(
                f"rendered prefix ({len(prefix_ids)}) plus longest question "
                f"({longest}) exceeds the {self.max_context}-token context"
            )
        answer_ids = [
            [self.token_id(t) for t in b.spec.answers] for b in rendered.branches
        ]
        jobs = []
        for n, (ids, branch_answers) in enumerate(zip(branch_ids, answer_ids)):
            wanted = sorted(set(branch_answers))
            for start in range(0, len(wanted), MAX_TOKEN_IDS):
                jobs.append(
                    (n, prefix_ids + ids, wanted[start : start + MAX_TOKEN_IDS])
                )
        workers = 1 if deterministic else self.concurrency
        with ThreadPoolExecutor(workers) as pool:
            results = list(pool.map(lambda job: self._complete(job[1], job[2]), jobs))
        logits: list[dict[int, float]] = [{} for _ in branch_ids]
        for (n, _, _), found in zip(jobs, results):
            logits[n].update(found)
        out = [
            [logits[n][t] for t in branch_answers]
            for n, branch_answers in enumerate(answer_ids)
        ]
        return out, len(prefix_ids) + sum(len(ids) for ids in branch_ids)
