# Nexus SmallDecision 26B A4B

An open-weight System One model built on Gemma 4 26B-A4B. It answers typed questions about a state in one forward pass and returns probabilities. This folder has the public Decision Index 0.3 result, the exact settings it was served with, the inference code (`nexus_systemone/`) and the points a reader should check before trusting the number.

## Public evaluation

Full public 0.3 suite with the kit at commit `25c6ec7`. All 140,178 scoreable requests were answered, nothing was truncated and no request was unsupported.

| Measure | Value |
|---|---|
| Decision Index | 53.54 |
| Raw index | 64.43 |
| Breadth index | 52.42 |
| Knowledge & Reasoning / Language / Retrieval / Tools / Arts (skill) | 39.65 / 58.15 / 60.01 / 69.04 / 36.23 |
| Latency, median / mean / p95 | 48.3 / 111.0 / 369.5 ms |
| Latency, p80 / p99 / max | 135.7 / 744.0 / 7,318.9 ms |

Latency is the kit's own HTTP measure: one request at a time, one RTX 5090, prompt construction included. Please measure latency yourselves.

- Results: [nexus-syntegra/nexus-smalldecision-26b-a4b-decision-index @ 2cae6192](https://huggingface.co/datasets/nexus-syntegra/nexus-smalldecision-26b-a4b-decision-index/tree/2cae619290e5b7213992dfa642d90f8dfc0fa7f0/runs/nexus-smalldecision-26b-a4b-v2.0). It holds the run directory in compact form, so re-scoring it with the kit gives the same index.
- Suite files: rebuilt locally, and `suite verify --edition 0.3` at `25c6ec7` matches every pinned hash. The SGD rows come from the builder at kit commit `f8552a1`, which predates the SGD fix `2d7791c`, because that build reproduces the published hash.
- The run used the stock `http` engine. The server answered to the model id `nexus-tinydecision-reserved-gemma4-26b-a4b-nvfp4-0.1.0`, an internal name that the release name replaced afterward.
- The run took from 2026-10-29 21:44 UTC to 2026-10-30 02:05 UTC.

## Model

- Weights: [nexus-syntegra/Nexus-SmallDecision-26B-A4B-v2.0-SystemOne-Jev](https://huggingface.co/nexus-syntegra/Nexus-SmallDecision-26B-A4B-v2.0-SystemOne-Jev), Apache-2.0. The repo is a single squashed commit, so its commit id changes whenever the card changes. The weight files are identified by the SHA-256 values in the repo's `SHA256SUMS`, and the Hub's own LFS hashes for the three weight files match them.
- Base: `nvidia/Gemma-4-26B-A4B-NVFP4` at revision `a19cfe00be84568a6867111c9a68c9c44fdcffe6`.
- What was trained: a uniform soup of four LoRA adapters, merged into the base at strength 0.75. The routed experts stay NVFP4. Attention, the dense MLP, the router, the embeddings and the merged LoRA are BF16.
- The shards the run served are byte-identical to the uploaded weight files.

## How it was served

Hardware: one RTX 5090 (32 GB) visible to the server, vLLM 0.30.0 (from the run's server log), one request at a time.

The run was served by the training repository's own runtime, which is not published. It started vLLM like this:

```bash
python -m src.tinydecision serve --base gemma4-26b-a4b-nvfp4 --source <merged weights> \
  --layout read_twice --engine vllm --vllm-batch-invariant --vllm-quantization none \
  --max-num-seqs 96 --max-model-len 32768 --port 8137
```

That command launches vLLM with `VLLM_BATCH_INVARIANT=1` and `VLLM_USE_FLASHINFER_SAMPLER=0` in the environment, and with `--max-model-len 32768`, `--max-num-seqs 96`, `--gpu-memory-utilization 0.85`, `--logprobs-mode raw_logits`, `--max-logprobs 1`, `--enable-prefix-caching` and `--enable-prompt-tokens-details`. Chunked prefill was on with `max_num_batched_tokens` 2,496 (vLLM raised it from its 2,048 default for this model). The vLLM log says it stores the KV cache as `fp8_e4m3`; no flag in the command asks for that.

`--layout read_twice` writes the state and the question twice before the answer slot. It applies when the shared prefix plus the longest once-read question branch is at most 3,000 tokens. Longer requests read once. On the public suite 97.55% of requests read twice. POP909 and API-Bank are always over the cap, so they read once.

## Running it yourselves

`nexus_systemone/` is a standalone System One server for this model. It needs only tokenizers, pydantic, fastapi, uvicorn, httpx, rfc8785 and anyio, and it talks to a vLLM server. Its README has the vLLM flags and the start commands. It serves `POST /v1/systemone` and `GET /v1/models`, so the kit's `http` engine works against it unchanged. It reads answers through vLLM and does not serve GGUF files.

The package is a port of the training runtime that served the public run. The rest of this section says how closely they agree, and the answers differ in places.

Comparison on a 704-row sample (every 200th row of the run) against the recorded run, counting rows whose top answer differs. Every comparison run below was made after the host's operating system was upgraded; the recorded run was made before it.

| Serving | Rows with a different top answer |
|---|---|
| Training runtime, served again over HTTP | 6 of 704 |
| This package over HTTP, on the release host's existing caches | 4, 5 and 5 of 704 (three starts); 5 with the training runtime's vLLM launch behind it |
| This package in-process against a vLLM server, no HTTP layer | 5 of 704 |
| The first run of this package's check | 21 of 704 (two later unchanged reruns gave 5 each; we do not know why the first differed) |
| This package with empty vLLM, Triton and inductor cache directories | 49, 49 and 48 of 704 |

- Prompts: for every row in the sample, and in a 621-row set described below, the package builds the same token ids as the training runtime for each shared prefix, each question branch and each answer-token list, and reports the same token usage.
- Repeat variation: two sweeps of one engine against one running server changed the top answer on 5 of 704 rows, for either engine.
- Compile caches: with the vLLM, Triton and inductor cache directories pointed at empty directories, the top answer changed on 48 to 49 of 704 rows (about 7%) against the recorded run. A start that reused the cache an empty-cache start had built stayed at 49. The three empty-cache runs agreed with each other within 4 rows. We did not identify the cause. A fresh machine will probably differ from the recorded run on a few percent of top answers, and we did not measure what that does to the Decision Index. The serving environment now has torch 2.13.0, triton 3.7.1, flashinfer-python 0.6.18.post1 and driver 615.71.09.
- Exact ties: 645 of the 323,259 recorded answers (0.2%), in 621 of 140,620 rows, are exactly uniform with confidence 0.0. The kit gives a tie to the first option in canonical order and counts it toward the score. We did not identify why those answers tie; close logits rounding to the same BF16 value is a guess we did not check. On those 621 rows, runs on the release host's existing caches changed the top answer on 24 to 34 rows (4 to 5%), and 603 to 611 of the 645 tied answers stayed tied. That set is chosen for ties, so those rates do not describe the whole suite.
- Not measured: a full public-suite run through this package.

## Read before trusting the number

- The model is zero-shot with one exception. 19 distinct tool-choice training items (30 trained instances over the four soup stages: 7 from the dataset, 12 from the teacher) took their text from BFCL function documentation, with no test cases or answers. No other Decision Index dataset, and no parent of one, appears in any mixture the soup's members trained on (checked for v8, v13, v14 and v15, and also for v18 and v19).
- The settings were chosen while reading public-suite scores, so 53.54 is not a clean held-out figure. The same model scored 48.07 at full LoRA strength with the state read once. Reading twice raised it to 51.48, and strength 0.75 on top of that gave 53.54.
- The score is the public suite only. We are not claiming a Full score.

## Private evaluation

Please run the private same-skill and new-domain evaluation to produce the Full score, and measure latency on your own hardware.

Training provenance, the data sources and the evaluation record are in the model card.
