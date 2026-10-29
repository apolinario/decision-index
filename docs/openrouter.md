# OpenRouter-first hosted evaluations

This working fork adds an `openrouter` engine. Dataset builders, suite editions,
scorers and weights are unchanged. The adapter calls
`https://openrouter.ai/api/alpha/decisions` with System One-shaped
`state` and keyed `questions`, **not** chat completions or OpenAI's native
`/v1/decisions` schema. Native OpenAI, TypeSafe and Inception adapters are deferred.

## Setup

```sh
uv sync --extra dev
uv run pytest -q
```

Set `OPENROUTER_API_KEY` in your shell or secret manager. The engine does not load
`.env` files. Never put a key in `--option`, a config, a result, or shell history.
No inference, model downloads or suite rebuilds occur during installation/tests.

## Models

Only these two profiles or their exact model IDs are allowed. Other hosted models
are rejected before creating a ledger or making a request:

| Profile | Model ID |
| --- | --- |
| `jev` | `typesafe/jev-1.14` |
| `mercury` | `inception/mercury-decide` |

These IDs were listed on 2026-10-27. Recheck availability and prices through the
[live catalogue](https://openrouter.ai/api/v1/models?output_modalities=decisions).
Mercury's `:free` route was not listed. Aliases containing `latest` are rejected.
Versioned IDs can still change behind the provider; native responses retain the
resolved model and provider where supplied. No fallback models or automatic
HTTP retries are configured.

## Small paired comparison

For a three-request connectivity smoke test **without downloading the suite**:

```sh
uv run decision-index run --engine openrouter --model jev \
  --rows examples/openrouter-smoke.jsonl --out runs/smoke-jev \
  --option allow_spend=true --option budget_usd=0.05 --option max_requests=3 \
  --option ledger=runs/smoke-jev/requests.jsonl
```

These are authored sanity checks, not official benchmark rows. Inspect the answers
in `results.jsonl`; do not compute a Decision Index from them. The upstream runner
only sends `state` and `questions`, never `expected` labels, to the engine.

First rebuild/import the suite following the root README (roughly 7 GB downloads;
some source terms require acceptance). Then sample once and reuse that exact file:

```sh
uv run decision-index suite sample --edition 0.3 --n 100 --out sample-100.jsonl.gz

uv run decision-index run --engine openrouter --model jev \
  --rows sample-100.jsonl.gz --out runs/jev \
  --option allow_spend=true --option budget_usd=0.25 --option max_requests=200 \
  --option ledger=runs/jev/requests.jsonl
```

The command above makes real remote requests. Run only after approving the spend
and the benchmark data leaving your machine. Use a separate output directory
**and ledger** for each model. Repeat with `--model mercury` and its own run/ledger.
Ollama Decider v1.1 Q4_K_M, GLiDE and StartLux Q4_K_M run
[locally](local-models.md), without OpenRouter credits.
Sampling preserves complete linked groups by default,
so `--n 100` may yield more than 100 requests. Increase limits deliberately.

Use the ordinary upstream `score` command with the same suite edition. Small or
incomplete runs are diagnostic, **not** comparable full public-index scores.
Neither this fork nor the public kit reproduces the board's private Full score.

## Spending and restart semantics

- Every route requires `allow_spend=true`, a positive
  `budget_usd`, `max_requests`, and a persistent ledger path.
- `max_requests` is a submission cap across resumes of that ledger, including
  failed requests. No paid warm-up requests are made.
- The dollar budget stops **before the next call** once observed cost reaches
  the threshold. **It is not a hard billing cap: the final request can overshoot.**
  Use an OpenRouter key/account spending limit for an external hard limit.
- Every submission is flushed to disk before sending. Responses (including
  malformed/error responses), HTTP timing, usage and reported cost are retained
  in the append-only ledger, even with upstream `--compact`.
- Missing cost is unknown, never zero. HTTP/transport failures and unknown costs
  halt the run. An unresolved submission blocks reopening the ledger: reconcile
  against provider usage before any retry. Do not erase the ledger or start a
  new one merely to bypass this protection. Automatic reconciliation is not yet
  implemented; preserve evidence and review the provider records manually.
- The ledger is exclusively locked against concurrent writers, and rejects a
  different model/configuration. Keep it with its corresponding results.
- Completed run IDs are checked against their original ledger payloads before
  resume skips them. Changed inputs require a separate run directory and ledger.
- Returned model IDs must match the requested profile or its explicitly pinned,
  observed provider revision. Unexpected IDs halt without retrying.
- Invalid probability vectors fail upstream validation without being repaired;
  their billed costs still count. As upstream documents, `error` rows can be
  retried on an explicit resume. The OpenRouter engine replays recorded responses
  by upstream run ID rather than paying again, including after a crash between
  ledger persistence and result persistence. Original HTTP timing is retained as
  `model_request_wall_ms`; `response_replayed` marks recovery. Use a fresh run and
  ledger only when deliberately commissioning a new measurement.
- No hidden request splitting, option filtering, truncation, recalibration or
  conversion of refusals into invented answers. Choice and Noul are supported;
  Score/Box/extraction are outside the current public text-suite engine.

All results and ledgers contain benchmark text. Keep them private under the
datasets' source licences. Credentials and Authorization headers are not logged.

Provider protocol normalization does not make models semantically identical:
question grouping, context support, calibration and usage accounting may differ.
Compare the same suite rows and disclose the resolved provider and runtime.
