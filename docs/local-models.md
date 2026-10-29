# Local models: Ollama Decider v1.1, GLiDE and StartLux

The matrix is five models: Jev 1.14 and Mercury Decide on OpenRouter;
Perplexity Decider **v1.1 Q4_K_M via Ollama**, Fastino GLiDE and StartLux Instant Q4_K_M self-hosted.
No other remote profiles are enabled. Local engines enforce loopback-only origins
and verify server/model identity: Ollama uses `/api/version` and `/api/tags`;
the other profiles use `/health`. None sends an OpenRouter credential.

Validation counts below describe local development checks. Referenced `runs/`
records are private local artifacts, not files distributed with this fork.

## Active Decider evaluation: Ollama v1.1 Q4_K_M

This replaces the previously planned v1.2 BF16 run. The downloaded v1.2 checkpoint
was removed at the user's request. Do not label the Ollama results v1.2 or compare
them as identical precision. The native v1.2 instructions below remain available
for a future run and require downloading the checkpoint again.

Use `--engine ollama-decider`, which calls `/v1/systemone` and checks the local
model's tag, quantization and full manifest digest before inference:

- Tag: `pplx-decider:27b-q4_K_M`.
- Manifest: `733585a9ebb6205d2d5b64a13748f0b1c9f1285885dd267780cfdb1c4c8ef141`.
- GGUF layer: `pplx-v1.1-Q4_K_M.gguf`, SHA-256
  `c5789d85610cd8b70be07e89164f3d2dd571c5ad5b1af65f2446f7598a329201`.

System Ollama was upgraded to **0.40.12** and serves on loopback port **11434**.
Use that installation for this project, not PAIR's separate managed runtime.
The isolated runtime download was stopped. System Ollama uses its existing store
`/usr/share/ollama/.ollama/models`; PAIR's user store is separate. Do not run
PAIR's proxy on the same port while evaluating.

The user pulled `pplx-decider:27b`; its full digest and Q4_K_M metadata match the
pinned model above. `ollama cp pplx-decider:27b pplx-decider:27b-q4_K_M` added the
explicit evaluation alias without duplicating or downloading model weights.

Smoke passed on 2026-10-28: three requests, four correct decisions, median warmed
HTTP time 56.84 ms, 100% GPU with a 16,384-token loaded context. Evidence:
`runs/smoke-ollama-decider-v1.1-q4-20261028/`. The model was unloaded afterward;
the system service remains available. This is not a benchmark ranking.

**Full-run stability finding:** the publisher's default `num_ctx=16384` caused
repeated HTTP 500 `Context size has been exceeded` failures in grouped BFCL
requests under system Ollama 0.40.12. The native runtime launches 16 internal
decision slots sharing the context. A fresh load let the first failed request
pass, but the error recurred later; those are runtime errors, not unsupported
inputs. The partial default-context run is preserved and paused.

A separate runtime-only alias `pplx-decider:27b-q4_K_M-ctx128k` sets
`num_ctx=131072` while reusing the identical backbone and projector blobs. Its
manifest digest is
`7002ddb908b18ed49c889f58b168af9e493d434ed1dfcc0426d03dc4f26c0ba6`.
The adapter supports it explicitly with `--option context=131072`; other context
values are rejected. The default remains unchanged. Do not mix results from the
two configurations. Bounded real-suite stability checks for this alias and
StartLux passed under `runs/local-readiness-20261028/`: Decider returned 1,769
valid responses and four explicit unsupported empty-state requests; StartLux
returned 1,773 valid responses with no errors. Both runtimes were unloaded/stopped
afterward. These are selected stability checks, not a full benchmark or proof that
every corpus input fits either runtime. Full benchmark runs remain paused.

The 128K Decider configuration completed all 1,694 BFCL stress requests, including
both requests that previously hit HTTP 500, without context errors. Additional
real-suite checks identified Ollama's rejection of empty string state (HTTP 400,
exact message `state must not be empty`) in Humicroedit and GPQA. These requests
carry their content in the questions and are valid benchmark rows. The adapter
records this specific native limitation as `Unsupported`, not a runtime error;
it does not invent state text, rewrite questions, or treat other HTTP 400 errors
as unsupported. Report the resulting coverage loss in benchmark scores.

```sh
# After upgrading and starting the normal Ollama server:
ollama pull pplx-decider:27b-q4_K_M
uv run decision-index run --engine ollama-decider \
  --rows examples/openrouter-smoke.jsonl --out runs/smoke-ollama-decider-v1.1-q4

# Separate 128K alias; identical weights, different context configuration:
curl --fail-with-body http://127.0.0.1:11434/api/create \
  -H 'Content-Type: application/json' \
  -d '{"model":"pplx-decider:27b-q4_K_M-ctx128k","from":"pplx-decider:27b-q4_K_M","parameters":{"num_ctx":131072},"stream":false}'
uv run decision-index run --engine ollama-decider --option context=131072 \
  --rows examples/openrouter-smoke.jsonl --out runs/smoke-ollama-decider-128k
```

The native Ollama decision implementation owns prompts, readout and calibration;
this is not chat generation with parsed labels. Server version/model metadata is
captured in run provenance. Request grouping is unchanged. Ollama limits local
text/schema to 64 KiB and requests to 64 questions; over-limit inputs are not
truncated or split by our adapter. Report unsupported coverage with scores.
Loading is excluded from warmed timing; no provider credits are spent.

## Separate serving environments

Use Python 3.12 and independent `uv` environments for the vendors' dependencies.
The eval environment stays lightweight (no Torch requirement). Each local model
in BF16 needs about 52 GB/GiB of weights plus workspace, fitting one 96 GB GPU for
ordinary contexts; the Q4 profiles use less memory. Use one GPU per model.
Long contexts can exceed 96 GB. Perplexity's published loader initially constructs
the backbone on CPU before moving it to GPU, so loading may use swap heavily.
GLiDE uses tensor-by-tensor GPU loading. GLiDE GPU inference was tested: a partial
full-suite run returned 2,299 valid responses before being stopped for slowness.
This is not a complete benchmark or a representative throughput estimate.

Download the full snapshots explicitly when ready; these commands transfer large
weights and vendor executable code. Nothing downloads at package install/test time:

```sh
# Run from the project root; these are the revisions inspected for this integration.
hf download perplexity-ai/pplx-decider-v1.2 --revision cce0f3f80033e9d89d80831d5281fd72124aacce --local-dir models/pplx-decider-v1.2
hf download fastino/GLiDE --revision d199b6d7e04523d87986b8b68cd62272ed1cefc0 --local-dir models/GLiDE

uv venv --python 3.12 .venv-pplx
uv pip install --python .venv-pplx/bin/python -r models/pplx-decider-v1.2/requirements.txt
uv pip install --python .venv-pplx/bin/python -e .

uv venv --python 3.12 .venv-glide
uv pip install --python .venv-glide/bin/python -r models/GLiDE/requirements.txt
```

Review the pinned vendor code/licences before executing it. Perplexity is
Apache-2.0; GLiDE uses the Fastino Community License. Current Hub metadata is
public, despite stale private-access wording in the model cards; authentication
may still be required depending on access policy. Don't substitute other revisions
silently. Keep checkpoint manifests/checksums with your results.

## Deferred: Perplexity-compatible v1.2 local endpoint

```sh
.venv-pplx/bin/python -m decision_index.pplx_server \
  --checkpoint models/pplx-decider-v1.2 --device cuda:0 --port 8011 \
  --max-length 32768
```

This binds **127.0.0.1 only** and exposes `POST /v1/decisions`, using Perplexity's
`model`, `state`, keyed `questions`, and keyed `answers` contract—not OpenAI's
Decisions schema. It accepts only the exact `pplx-decider-v1.2` model ID.

The original `pplx_decider.Decider.decide` builds the joint prompt, uses the
noncausal/per-marker readouts and applies the saved temperature once. We do not
replace it with ordinary causal chat generation or a hand-written prompt.
Question grouping/order and native probabilities are retained.

Supported **text/JSON subset** of the [Perplexity API](https://docs.perplexity.ai/docs/decisions/quickstart):

- 1–128 questions; Choice 1–255 options; Noul; Score 2–10 ordered levels.
- Structured state stays JSON; unknown top-level/question fields are rejected.
- `score` is expected level; summary confidence follows the hosted API's
  documented formula (the downloadable runtime uses a different score baseline).
  `output_tokens` is the question count, as on the hosted API; input tokens come
  from the native runtime. No fabricated provider billing/cost is reported.
- JSON errors and validation errors return 400 with `error.type/message`;
  body limit is 32 MiB; responses carry `x-request-id`.
- Images and single-level Score are explicitly unsupported, not silently turned
  into text or different questions. API keys, hosted rate limiting and billing
  aren't replicated. This is an evaluation server, not a production API clone.

`--max-length 32768` deliberately stays within training length and GPU headroom.
The author's runtime recursively splits questions if necessary; the health endpoint
records this limit. The hosted service has a larger context and its grouping
algorithm is not byte-identical, so do not claim exact hosted/local parity.
`--fast` opts into the author's compiled path; install its `requirements-fast.txt`
first and record that setting. Reference mode is the default.

## GLiDE's native server

Use the checkpoint's own server rather than a reimplementation:

```sh
PYTHONPATH="$PWD/models/GLiDE" GLIDE_DEVICE=cuda:1 GLIDE_MAX_BATCH=1 \
  .venv-glide/bin/python -m glide.server \
  --model "$PWD/models/GLiDE" --host 127.0.0.1 --port 8012
```

Install the vendor-recommended optional `flash-linear-attention` in that
environment for validated performance. The API is `POST /v1/systemone`.
One question is one native forward pass; no request grouping is changed here.
The single-batch setting reduces probability variation from concurrent traffic.

## Evaluate the same inputs

In the lightweight eval environment, no API credits or cloud key are needed:

```sh
uv run decision-index run --engine pplx-decider \
  --rows examples/openrouter-smoke.jsonl --out runs/smoke-local-decider
uv run decision-index run --engine glide \
  --rows examples/openrouter-smoke.jsonl --out runs/smoke-local-glide
```

Replace `--rows` with the same group-preserving public-suite sample used for Jev
and Mercury. Do not pass `--model` to the local profiles: they have fixed model
IDs. You may override the loopback origin via `--option base_url=http://127.0.0.1:PORT`.
Health metadata is captured in provenance before running; local HTTP latency
includes inference/queueing but excludes loading. Upstream warmup uses two real
local requests. The text-suite validator/scorer remains Choice/Noul-only even
though the local server also exposes Score.

Self-hosted means no provider credits, not zero electricity/hardware cost. Do not
expose either server to a network: the Perplexity subset has no authentication,
TLS, hostile-client timeouts or production resource isolation.

## StartLux Instant Q4_K_M (text only)

Requested file: `StartLux-Decision-v1.1-27B-Instant-Q4_K_M.gguf` in
`models/StartLux-27B-Instant-GGUF`, Hub revision
`67834e921e8a6f977088d9b560ee9e02dffc87bc` (approximately 16.55 GB).
Weights are **CC BY-NC 4.0**, not a commercial-use grant. The author's inference
package is Apache-2.0. No Q8 weights or vision projector are required here.

Tokenizer/config snapshot: `models/StartLux-27B-Instant-config`, revision
`32e7d35d74db615c4a059181efbf21c5421f3561`, excluding safetensors.
Use the author's `startlux-decide` wrapper at
`a99977d9332decc9fb27a2488b4a83b533b6cd5f` with llama.cpp `b11549`
(`aa94f2086147cd4bfbcff818ce998dffce5fcb39`). Tool sources/builds live in ignored
`.tools/`; Python dependencies are isolated in `.venv-startlux`.

After a CUDA build of llama-server, start these two processes in separate terminals
(choose a free GPU; do not compete with a benchmark already running):

```sh
CUDA_VISIBLE_DEVICES=0 .tools/llama.cpp-b11549/build/bin/llama-server \
  -m models/StartLux-27B-Instant-GGUF/StartLux-Decision-v1.1-27B-Instant-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8081 --parallel 1 --ctx-checkpoints 0 \
  --override-kv qwen35.attention.causal=bool:false \
  -c 20480 -b 16384 -ub 16384 -ngl 99

STARTLUX_LLAMACPP_BIDIR=server .venv-startlux/bin/startlux-decide serve \
  --model models/StartLux-27B-Instant-config --backend llamacpp \
  --llamacpp-url http://127.0.0.1:8081 --host 127.0.0.1 --port 8090 \
  --name startlux-decision-v1.1-27b-instant-q4-k-m --no-images --max-model-len 16384

uv run decision-index run --engine startlux \
  --rows examples/openrouter-smoke.jsonl --out runs/smoke-local-startlux
```

Instant requires noncausal full-attention layers and a whole-prompt micro-batch;
ordinary causal chat serving is not equivalent. Prompts must fit the 16,384-token
micro-batch, even though the server context is larger. The wrapper owns prompt
construction and saved per-type temperature calibration. Its default internal
HTTP policy allows **four attempts** for transient backend failures; client-side
evaluation adds no retries, but measured latency includes those internal attempts.
Do not label this runtime no-retry. Author defaults (including its internal
none-of-the-above check) are retained and exposed in health provenance.

The profile verifies the explicit served name, not the GGUF hash: retain checkpoint
integrity records separately.

GPU smoke inference passed on 2026-10-27: three fixture requests, four correct
decisions, median warmed client HTTP time 127.32 ms. This is a connectivity and
schema check, not a benchmark ranking. Evidence is in
`runs/smoke-startlux-q4-20261027/`; GPU 0 total usage was approximately 23 GiB.
The wrapper needed `uv pip install --python .venv-startlux/bin/python jinja2==3.1.6`
because its dependency set did not install the chat-template dependency. The
tokenizer's option-letter checks pass. Build uses CUDA 13.4, architecture 120a;
NCCL is absent, which does not affect this single-GPU setup.
