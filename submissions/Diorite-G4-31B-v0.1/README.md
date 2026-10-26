# Diorite-G4-31B-v0.1

- Model: [`Leucodiorite/Diorite-G4-31B-v0.1`](https://huggingface.co/Leucodiorite/Diorite-G4-31B-v0.1/tree/0fcbaa5fc3dc255aa2c693115a31f517cac247ff) (gated, manual approval; multimodalart has access).
- Kind: PEFT LoRA adapter on stock `google/gemma-4-31B-it` @ `842da379`, loaded unmerged. r 64, alpha 64, dropout 0; q/v/o and gate/up/down projections of the language model (350 modules; no k_proj, vision, audio or embedding weights).
- Edition 0.3, full public suite. `scores.json` has `"complete": true`: 140,178 of 140,178 scored requests ok.
- Decision Index **60.26** (raw 69.63, breadth 59.34). Areas: knowledge 47.54 · language 63.38 · retrieval 66.80 · tools 75.88 · arts 43.40.
- The original gemma-4-31B-it, run without the adapter through the same batch runner and prompt layout: 57.39.

## Serving

`diorite_serve` in the model repo is a FastAPI server with in-process vLLM that answers `POST /v1/systemone`.
- Prompt layout: the Diorite prompt layout. One prompt per question.
- It reads every option's probability from one forward pass and generates nothing.
- The adapter is applied per request through vLLM LoRA; a request that names `base` (or the base model's own name) reads the base weights instead.

```
python -m diorite_serve --model google/gemma-4-31B-it --layout chat --adapter diorite-g4-31b-v0.1=./adapter --default_model diorite-g4-31b-v0.1 --max_model_len 65536 --max_batched_tokens 4096 --gpu_memory_utilization 0.90 --port 8000
decision-index pipeline --engine http --option base_url=http://127.0.0.1:8000 --option model=diorite-g4-31b-v0.1 --out runs/diorite-g4-31b-v0.1
```

Activate the environment before starting the server (or put its `bin/` on `PATH`): vLLM compiles a few sampler kernels with `ninja` on first use.

Settings:
- bf16 weights and KV cache, vLLM 0.29.0 with its default attention backend.
- LoRA unmerged (max_lora_rank 64, max_loras 1).
- max_model_len 65,536, max_batched_tokens 4,096, gpu_memory_utilization 0.90. They start on one RTX PRO 6000 (96 GB).
- temperature 1.0 for every question type, read once.
- No other inference techniques: no re-reading (diorite-serve's optional read-twice mode stays off), no second reading of unsure questions, no ensembling, no thinking, and no per-benchmark prompts or temperatures.

Over-long prompts are refused with HTTP 413, which the kit records as unsupported.

## How the submitted rows were made

- **Runner.** Rows were produced by our batch runner in the Diorite prompt layout (vLLM 0.29.0, LoRA adapter on the base weights, temperature 1, read once), converted to the kit format and scored with kit f5a6e4fd. diorite-serve 0.2.0, the server we submit, renders the same token ids (checked with the google/gemma-4-31B-it, google/gemma-4-12B-it and google/gemma-4-E4B-it tokenizers) and reproduces the submitted decisions on the kit's 750-row sample: 6,599 of 6,616 questions (99.74%) get the same answer, on one RTX PRO 6000 through the kit's `http` engine, one request at a time (per question, the largest change in any option's probability: median 2.7e-5, max 0.15). That check ran on a release candidate of diorite-serve 0.2.0. On 108 rows drawn from that sample, a later build with the same prompt construction and engine settings, started with the model card's command on one RTX PRO 6000 and driven by the kit's `http` engine, reproduces 979 of 980 submitted decisions, and its offline reproduction script also 979 of 980. The released build differs from it only in input validation, error handling and tools, and its prompts were re-checked token for token.
- **Runner details.** Our batched offline runner, not the kit's serial runner: vLLM 0.29.0, transformers 5.20.2, torch 2.13.0+cu130, bf16, max_model_len 40,960. The longest public prompt is 32,316 tokens, so nothing was refused.
- **Hardware.** Two shards, one NVIDIA H200 each.
- **Scoring.** The outputs were written in the kit's result format and scored with `decision-index score --edition 0.3` at `f5a6e4fd`. Request times were not recorded.

`tools/reproduce_public_run.py` in the model repo's diorite-serve answers the suite's request files offline the same way as the runner (paths are arguments); see `PROVENANCE.md` in the results dataset.

Request time on the kit's 750-row sample (`decision-index suite sample --n 750`; one RTX PRO 6000, one request at a time; the sample weights every benchmark equally): median 150 ms, mean 519 ms, p80 1,080 ms. The slowest rows are long or many-question requests (BRIGHT, API-Bank, POP909, ToolRet, ACOS). Reweighted to the per-benchmark counts of your latency-v1 sample, the same rows give about median 99–119 ms, mean 281–304 ms, p80 295–424 ms; the range covers how we fill in the 50 ForecastBench rows of your sample, a benchmark the 0.3 suite no longer runs. Self-measured, for reference only.

## Training data

Mostly the training splits of public multiple-choice and other decision datasets. Every training item that also appears in the leaderboard's public suite was removed.

## Declared limits

The server refuses prompts of 65,536 tokens or more as unsupported. Nothing is truncated and no option is removed.
