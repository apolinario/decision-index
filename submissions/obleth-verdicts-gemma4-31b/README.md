# obleth verdicts gemma4-31b

- Model: stock [`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it/tree/842da3794eaa0b77d5f08bae87a17459d91ff475) @ `842da379`. No fine-tuning, no adapter: this entry is an inference technique on the original weights.
- Engine: the `/v1/systemone` endpoint of the [obleth](https://github.com/thediymaker/obleth-gateway) gateway at commit [`6efdc08`](https://github.com/thediymaker/obleth-gateway/commit/6efdc08cc7f777c0dddfa6c5560d9182c7c73852), in front of vLLM.
- Results: [`runs/obleth-verdicts-gemma4-31b`](https://huggingface.co/datasets/562uned/obleth-verdicts-gemma4-31b-decision-index/tree/ec9f0d3050248a270638fa555186d264cba378f3/runs/obleth-verdicts-gemma4-31b) (gated, manual approval).
- Edition 0.3, full public suite. `scores.json` has `"complete": true`: 140,178 of 140,178 scored requests ok.
- Decision Index **57.50** (raw 67.72). Areas: knowledge 44.38 · language 61.50 · retrieval 62.05 · tools 73.56 · arts 42.55.

## How it answers

`/v1/systemone` translates each request into obleth's verdicts format and reads every answer from one forward pass per question. It generates nothing beyond one token.

- **One call per question.** Each question is one chat call with `max_tokens: 1`, `temperature: 0`, `logprobs: true` and `top_logprobs: 20`.
  - The state sits in a system message that is identical for every question of a request, so the backend's prefix cache processes it once.
  - Questions are sent in chunks of 32.
- **Labels.** Options are labeled in sorted-name order: `A`–`Z` for up to 26 options, two-letter codes `AA`–`JU` for 27–255. `noul` questions are asked as yes/no. Score levels are labeled `A`, `B`, ….
- **Reading the answer.** The first token's probabilities over the labels are renormalized into the answer's distribution.
  - For a tokenizer that splits a two-letter code, a second single-token call continues after each first letter with at least 1% of the mass, giving P(code) = P(first) × P(second | first). Gemma's tokenizer emits most codes as one token; this run made continuation calls on 333 requests (332 POP909-CL, 1 ChessBench).
- **No answer.** If the model puts under 5% of its mass on the labels, the question is answered with an even distribution and confidence 0 (`answered: false`). This happened on 4 questions (2 MMLU, 2 MMLU-Pro).
- **No other techniques:** no re-reading, no ensembling, no thinking, no per-benchmark prompts.

## Serving it on one GPU

Start vLLM and the gateway on the same Docker network, register the model, make a key, and point the kit at the gateway:

```sh
git clone https://github.com/thediymaker/obleth-gateway && cd obleth-gateway && git checkout 6efdc08
cd deploy/docker && cp .env.example .env
docker compose up -d --build obleth
docker run -d --gpus all --network obleth_default --name vllm vllm/vllm-openai:v0.29.0 \
  --model google/gemma-4-31B-it --revision 842da3794eaa0b77d5f08bae87a17459d91ff475 \
  --served-model-name gemma4-31b-it --max-model-len 65536 --enable-prefix-caching

A=http://127.0.0.1:9180; H="Authorization: Bearer dev-admin-token"   # OBLETH_ADMIN_TOKEN from .env
curl -sf -H "$H" -H 'Content-Type: application/json' $A/api/v1/models \
  -d '{"model_name":"gemma4-31b-it","upstream_model":"gemma4-31b-it","api_base":"http://vllm:8000/v1","model_type":"chat","context_window":65536}'
T=$(curl -sf -H "$H" -H 'Content-Type: application/json' $A/api/v1/tenants -d '{"name":"decision-index"}' | jq -r .id)
export DECISION_INDEX_API_KEY=$(curl -sf -H "$H" -H 'Content-Type: application/json' $A/api/v1/tenants/$T/keys -d '{"name":"decision-index"}' | jq -r .secret)

decision-index pipeline --engine http --option base_url=http://127.0.0.1:8088 --option model=gemma4-31b-it --out runs/obleth-verdicts-gemma4-31b
```

The gateway itself needs no GPU. Its settings for this model are the defaults: no reasoning options are sent, so Gemma answers without thinking.

## How the submitted rows were made

- **Runner.** The kit's own runner at `25c6ec7` with the stock `http` engine, posting each request unchanged to `/v1/systemone`.
- **Sharding.** The kit's runner sends one request at a time, so the suite was split into 8 shards, each a `decision_index.runner.run` call with a `keep` filter on crc32 of the case group (run id when there is none), so linked cases stay in one shard.
- **Merging and scoring.** The shards' `results.jsonl` files were concatenated and scored with `decision-index score --edition 0.3` at `25c6ec7`.
- **Retries.** 7 requests hit transient HTTP 502s and were retried by the runner's resume. Their error rows stay in the file ahead of the ok rows; the scorer keeps the last row per run id, as resume does.
- **Hardware.** 4 × Intel Gaudi2 (tensor parallel 4) per model replica, vLLM for Gaudi (v0.26 line), bf16.
  - The backend was shared with other traffic during the run.
  - Its one config change during the run (vLLM `--reasoning-parser`, which only affects how generated text is split) was checked on 442 rows. Answers before vs after agree on 99.68% of questions, and the same rows read twice afterwards agree on 99.88%. Gaudi batching noise is of that size.

Request time in this run (all 140,620 requests, wall time at the client, 8 shards in parallel on a shared backend): median 209 ms, mean 507 ms, p80 719 ms. Self-measured, for reference only.

## Declared limits

None in the public suite. The submitted run served a 131,072-token window, and its longest prompt was about 40,500 tokens, so the 65,536 in the command above also covers it. A prompt over the window is refused with HTTP 422 `maximum context length`, which the kit records as unsupported. Nothing is truncated and no option is removed.
