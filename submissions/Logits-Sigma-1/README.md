# Logits Sigma-1

[logitshq/Logits-Sigma-1](https://huggingface.co/logitshq/Logits-Sigma-1/tree/5ee69b5ace03c6bdd4e3391ac722993baa7c8d2d)
is google/gemma-4-31B-it (revision `842da379`) with one LoRA fine-tune (rank 64) merged into the bf16 weights at
scale λ = 0.75 (0.75 · fine-tuned + 0.25 · base), Apache-2.0. It returns the probability of each option's one-token
code at the answer position, one prefill per question; it generates nothing.

| | |
|---|---|
| Decision Index 0.3 public index | **67.1** (raw 75.0); `score --edition 0.3` on the published `results.jsonl.gz` with the kit at `25c6ec7` gives the same |
| Requests | complete: 140,178 of 140,178 scoreable requests answered, 0 unsupported, nothing truncated, 0 errors |
| Request time, one request at a time | median 122, mean 356, p80 308 ms over the latency-v1 design (750 timed rows + 10 warm-up, benchmark × length, seed 20260926) on one NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB), with the options below |
| Request time in `scores.json` | median 67.6, mean 209.9 ms: eight engine processes, one per H200, so not an RTX PRO 6000 figure |
| Model | [`logitshq/Logits-Sigma-1`](https://huggingface.co/logitshq/Logits-Sigma-1/tree/5ee69b5ace03c6bdd4e3391ac722993baa7c8d2d): public; every file matches `SHA256SUMS` there |
| Results | [`runs/logits-sigma-1-kvfp8`](https://huggingface.co/datasets/logitshq/logits-sigma-1-decision-index/tree/51fe749c15ab5a61e7c91237c91c6ef80559b4ed/runs/logits-sigma-1-kvfp8) @ `51fe749c`: `results.jsonl.gz` (compact: no suite text), `scores.json`, `index.json`, `benchmark-summary.json`, `environment*.json` (one per shard), `release_check.json`, `rescore/` |

## Run

```sh
hf download logitshq/Logits-Sigma-1 --revision 5ee69b5ace03c6bdd4e3391ac722993baa7c8d2d --local-dir logits-sigma-1
uv venv --python 3.12 venv && . venv/bin/activate
uv pip install "vllm==0.33.0" ./logits-sigma-1/code          # torch 2.14.0, transformers < 5.21
uv pip install --no-deps "decision-index @ git+https://github.com/apolinario/decision-index@25c6ec7"
python -m decision_index pipeline --engine logits_decision.engine_vllm:VllmEngine \
    --option model=logits-sigma-1 --option kv_cache_dtype=fp8 --compact --out runs/logits-sigma-1
```

One 96 GB GPU holds the model: the bf16 weights take ~58 GiB, and vLLM's fp8 KV cache (`kv_cache_dtype=fp8`) leaves
room for 131,072-token prompts next to them. With a bf16 cache it would not start on an RTX PRO 6000 (22.7 GiB of cache
left, 35.8 GiB needed). These exact options started and passed the latency gate on one RTX PRO 6000; no attention
backend override was needed there.

## Engine

`logits_decision.engine_vllm:VllmEngine`, shipped in the model repo under `code/` (Apache-2.0).

- **In-process vLLM 0.33.0**, merged bf16 weights, fp8 KV cache, text only (`language_model_only`), prefix caching on,
  CUDA graphs.
- **Prompt** (layout v5, read from the checkpoint's `merge.json`), one per question, in Gemma 4's chat template with
  thinking off: the instruction; "Questions about this state:" and every question of the request, numbered, in
  request order (question text only); `State:` and the state (a JSON state as JSON); then `Question:` with this
  question and `Options:` labelled `A.`, `B.`, … (then two-letter codes that are single tokens after `Answer:`, up to
  255). The model turn starts with `Answer:`.
- **Shared block, read twice:** everything before `Question:` depends only on the request, so a request's prompts share
  it as one token prefix, which vLLM's prefix cache computes once. When that block is at most 3,000 tokens it is written
  a second time after "Read the questions and the state again."
- **Readout:** `SamplingParams(max_tokens=1, temperature=0, allowed_token_ids=<the question's codes>,
  logprobs=<n codes>)` with `logprobs_mode="processed_logits"`: the codes' logits after Gemma's final softcap, divided by
  the temperature (1.0) and softmaxed over the question's codes. The one generated token is discarded. A `noul`
  question is a two-option choice (its criteria text, else Yes / No), reported as p(yes).
- **Several questions:** all of a request's prompts go to vLLM in one `generate` call; when they share at least 512
  leading tokens, the first is prefilled alone so the others reuse its cached prefix.
- **Never:** truncation, option dropping, per-benchmark prompts, labels, retries on a wrong answer.

Options, exactly as run:

```text
--engine logits_decision.engine_vllm:VllmEngine
--option model=<local copy of logitshq/Logits-Sigma-1>
--option kv_cache_dtype=fp8
--compact
```

Defaults set by the engine: `temperature` 1.0, `twice_max_tokens` 3000, `max_len` 131072 (`max_model_len` 131,073),
`gpu_memory_utilization` 0.90, `enable_prefix_caching` true, `share_min_tokens` 512, `logprobs_mode`
processed_logits, `seed` 0.

## Declared limits

Up to 131,072 prompt tokens per question and 255 options per question; a longer prompt is refused (`Unsupported`),
never truncated. No limit on questions per request. In the public suite no request was refused.

## How the run was made

- Suite: rebuilt with the kit at `9eb2dbe` (the 0.3 SGD builder; `main` changed it afterwards, see the kit's issue
  #274) and verified byte-identical (release-v1 `288d3720…`, v2 cut `b2b56d6f…`, added rows `7429f3c9…`, GSM8K
  `75933489…`).
- Run: a fresh run with the kit's runner at `25c6ec7` and the options above: the scoreable rows split into 8 shards by
  estimated cost, one engine process per H200, one request at a time per process, `--compact`; the shards were
  concatenated (one result per run id) and scored once with `score --edition 0.3`. Nothing failed and nothing was
  retried.
- Agreement (750 latency-v1 rows, 1,663 decisions): the served engine with a bf16 cache agrees with the unmerged
  adapter in Transformers on 99.16%, and a second vLLM run on 100%; with the fp8 cache on the RTX PRO 6000 it agrees
  with the H200 bf16-cache run on 99.04% and with the unmerged adapter on 99.16%.

## Training data and decontamination

476,186 training rows in 325,537 requests (480.3M tokens), one epoch planned, about 78% trained:

- our v5 set (411,332 rows): tasksource-jev-typed-decisions (411 sources tagged for commercial use), train splits of
  ten Decision Index benchmarks in their request format (BANKING77, CLINC150, HellaSwag ActivityNet contexts,
  WinoGrande, ARC, CommonsenseQA, GSM8K, ESCI, ContractNLI, iSarcasmEval; 38,278 rows), non-benchmark exam and QA sets,
  tool-calling and retrieval sets, procedural generators of our own, and decision rows built from public US records
  (NHTSA, OSHA, FJC courts, openFDA, CVE / CWE / CISA KEV, USPTO) and the Public Jira Dataset;
- 64,862 teacher rows (13.6%): task families designed by Kimi K3.1 Pro, states written by DeepSeek V4.1 Flash, labels
  from both models' votes, all on our own fleet. No text or label comes from a Claude or OpenAI model, or from any
  decision model.

No test or validation split of a Decision Index source dataset was read. Every training row was checked against the
0.3 public suite and the evaluated splits by 13-gram overlap and exact match (`decision/decontam.py`); a final pass
dropped 9,773 rows, and the generators dropped more at build time. The run was promoted, and λ = 0.75 chosen, on our
held-out families, our new-domain set and a fixed 10% subset of the public suite (14,956 requests); the λ rule was
fixed before its numbers.

Licence disclosure: the weights are Apache-2.0. The training data includes CC BY-SA sources and 6,133 rows (1.3%) from
12 sources labelled GPL / AGPL / ODbL / MPL or possibly non-commercial upstream; each source, its published licence and
its row count are in the model repo's `DATA_SOURCES.md`.
