# Diorite-G4-12B-v0.1

- Model: [`Leucodiorite/Diorite-G4-12B-v0.1`](https://huggingface.co/Leucodiorite/Diorite-G4-12B-v0.1/tree/13c07f31dc9d07ae2b8e076cf409be8134ebeb3c) (gated, manual approval; multimodalart has access).
- Kind: PEFT LoRA adapter on stock `google/gemma-4-12B-it` @ `707f0a3b`, loaded unmerged. r 64, alpha 64, dropout 0; q/v/o and gate/up/down projections of the language model (280 modules; no k_proj, vision, audio or embedding weights).
- Edition 0.3, full public suite. `scores.json` has `"complete": true`: 140,178 of 140,178 scored requests ok.
- Decision Index **53.42** (raw 64.61, breadth 51.71). Areas: knowledge 34.47, language 59.39, retrieval 62.10, tools 71.43, arts 36.64.
- The original gemma-4-12B-it, run without the adapter through the same batch runner and prompt layout: 48.45.

## Serving

Same server and settings as Diorite-G4-31B-v0.1, with the 12B base:

```
python -m diorite_serve --model google/gemma-4-12B-it --layout chat --adapter diorite-g4-12b-v0.1=./adapter --default_model diorite-g4-12b-v0.1 --max_model_len 65536 --max_batched_tokens 4096 --gpu_memory_utilization 0.90 --port 8000
decision-index pipeline --engine http --option base_url=http://127.0.0.1:8000 --option model=diorite-g4-12b-v0.1 --out runs/diorite-g4-12b-v0.1
```

Activate the environment before starting the server (or put its `bin/` on `PATH`): vLLM compiles a few sampler kernels with `ninja` on first use.

- Prompt layout: the Diorite prompt layout. One prompt per question.
- bf16 weights and KV cache, vLLM 0.29.0 with its default attention backend; LoRA unmerged (max_lora_rank 64, max_loras 1); temperature 1.0 for every question type, read once.
- No other inference techniques: no re-reading (diorite-serve's optional read-twice mode stays off), no second reading of unsure questions, no ensembling, no thinking, and no per-benchmark prompts or temperatures.
- Over-long prompts are refused with HTTP 413, which the kit records as unsupported.

## How the submitted rows were made

- **Runner.** Rows were produced by our batch runner in the Diorite prompt layout (vLLM 0.29.0, LoRA adapter on the base weights, temperature 1, read once), converted to the kit format and scored with kit f5a6e4fd. diorite-serve 0.2.0, the server we submit, renders the same token ids (checked with the google/gemma-4-31B-it, google/gemma-4-12B-it and google/gemma-4-E4B-it tokenizers) and reproduces the submitted decisions on the kit's 750-row sample: 6,589 of 6,616 questions (99.59%) get the same answer, on one RTX PRO 6000 through the kit's `http` engine, one request at a time (per question, the largest change in any option's probability: median 3.6e-5, max 0.23). That check ran on a release candidate of diorite-serve 0.2.0. On 108 rows drawn from that sample, a later build with the same prompt construction and engine settings, started with the model card's command on one RTX PRO 6000 and driven by the kit's `http` engine, reproduces 980 of 980 submitted decisions, and its offline reproduction script 979 of 980. The released build differs from it only in input validation, error handling and tools, and its prompts were re-checked token for token.
- **Runner details.** Our batched offline runner, not the kit's serial runner: vLLM 0.29.0, transformers 5.20.2, torch 2.13.0+cu130, bf16, max_model_len 40,960. Nothing was refused.
- **Hardware.** One NVIDIA H200.
- **Scoring.** Written in the kit's result format and scored with `decision-index score --edition 0.3` at `f5a6e4fd`. Request times were not recorded.
- **LoRA readout check.** The runner's vLLM LoRA readouts match a PyTorch + PEFT reference on all 316 decidable rows of a 320-row check (4 exact ties excluded), median |dp| 0.0054, p95 0.038.

`tools/reproduce_public_run.py` in the model repo's diorite-serve answers the suite's request files offline the same way as the runner (paths are arguments); see `PROVENANCE.md` in the results dataset.

Request time on the kit's 750-row sample (`decision-index suite sample --n 750`; one RTX PRO 6000, one request at a time; the sample weights every benchmark equally): median 64 ms, mean 213 ms, p80 433 ms. Reweighted to the per-benchmark counts of your latency-v1 sample, the same rows give about median 48 ms, mean 119 ms, p80 129 ms, leaving out the 50 ForecastBench rows of your sample, a benchmark the 0.3 suite no longer runs. Self-measured, for reference only.

## Training data

The same as Diorite-G4-31B-v0.1: mostly the training splits of public multiple-choice and other decision datasets. Every training item that also appears in the leaderboard's public suite was removed.

## Declared limits

The server refuses prompts of 65,536 tokens or more as unsupported. Nothing is truncated and no option is removed.
