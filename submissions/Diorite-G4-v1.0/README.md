# Diorite-G4-v1.0 (31B, 12B, E4B, E2B)

- Models (public, not gated): [`Leucodiorite/Diorite-G4-31B-v1.0`](https://huggingface.co/Leucodiorite/Diorite-G4-31B-v1.0/tree/24f3d454ab336a20a495e664fe9b250716d87270), [`Leucodiorite/Diorite-G4-12B-v1.0`](https://huggingface.co/Leucodiorite/Diorite-G4-12B-v1.0/tree/3490ec5e390e126d3862177cfc3e751ecd0f31a9), [`Leucodiorite/Diorite-G4-E4B-v1.0`](https://huggingface.co/Leucodiorite/Diorite-G4-E4B-v1.0/tree/47bcde52f33d183a3c49825802c40e4dfe6f7ff0), [`Leucodiorite/Diorite-G4-E2B-v1.0`](https://huggingface.co/Leucodiorite/Diorite-G4-E2B-v1.0/tree/b91d6b85eb9f3be923a6b85f0a2e171e3a9a86e6).
- Results (public, not gated): `Leucodiorite/Diorite-G4-<size>-v1.0-decision-index`, linked per size in `submissions/README.md`.
- Kind: PEFT LoRA adapters on the stock `google/gemma-4-<size>-it` models, loaded unmerged. r 64, alpha 64, dropout 0; q/v/o and gate/up/down projections of the language model (no k_proj, vision, audio or embedding weights).
- Edition 0.3, full public suite. Every `scores.json` has `"complete": true`: 140,178 of 140,178 scored requests ok.

| Size | Base revision | Index | Raw | Breadth | Knowledge | Language | Retrieval | Tools | Arts | Base model alone |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 31B | `842da379` | **65.47** | 73.72 | 64.81 | 51.97 | 67.67 | 69.57 | 80.82 | 58.41 | 57.39 |
| 12B | `707f0a3b` | **57.37** | 67.83 | 56.00 | 38.01 | 61.40 | 64.23 | 75.50 | 50.14 | 48.45 |
| E4B | `ee0ef602` | **51.27** | 63.53 | 49.33 | 28.51 | 58.98 | 56.15 | 68.77 | 48.39 | 38.01 |
| E2B | `3e22461f` | **43.17** | 57.53 | 40.62 | 20.06 | 49.05 | 50.77 | 63.58 | 35.20 | 25.91 |

"Base model alone" is the stock base model evaluated on its own, without the adapter, with the same prompts and the same token count per request.

## Serving

The same server and flags for every size, with that size's base model and adapter name (31B shown):

```
python -m diorite_serve --model google/gemma-4-31B-it --layout chat --adapter diorite-g4-31b-v1.0=./adapter --default_model diorite-g4-31b-v1.0 --max_model_len 65536 --max_batched_tokens 4096 --gpu_memory_utilization 0.90 --port 8000
decision-index pipeline --engine http --option base_url=http://127.0.0.1:8000 --option model=diorite-g4-31b-v1.0 --out runs/diorite-g4-31b-v1.0
```

Activate the environment before starting the server (or put its `bin/` on `PATH`): vLLM compiles a few sampler kernels with `ninja` on first use.

- Prompt layout: the Diorite prompt layout. One prompt per question.
- diorite-serve 0.3.1, in each model repo. bf16 weights and KV cache, vLLM 0.29.0 with its default attention backend; LoRA unmerged (max_lora_rank 64, max_loras 1 with one adapter); temperature 1.0 for every question type, read once.
- No other inference techniques: no re-reading (diorite-serve's optional read-twice mode stays off), no second reading of unsure questions, no ensembling, no thinking, and no per-benchmark prompts or temperatures.
- Over-long prompts are refused with HTTP 413, which the kit records as unsupported.
- The server also has an OpenAI-compatible `/v1/chat/completions` route on the base weights; the kit does not use it.

## How the submitted rows were made

- **Runner.** Rows were produced by our batch runner in the Diorite prompt layout (vLLM 0.29.0, transformers 5.20.2, torch 2.13.0+cu130, bf16, LoRA unmerged on the base weights, temperature 1, read once, max_model_len 40,960), converted to the kit format and scored with `decision-index score --edition 0.3` at `f5a6e4fd`. The runner builds the prompts and formats the answers with diorite-serve's own `wire.py` and `layout.py`. The longest public prompt is 32,316 tokens, so nothing was refused. Request times were not recorded.
- **Hardware.** NVIDIA H200, one GPU per shard: 31B in 4 shards, 12B in 2, E4B and E2B in 1. The shards were concatenated and scored once.

**Agreement with the submitted server.** diorite-serve 0.3.0 on one RTX PRO 6000 (96 GB) through the kit's `http` engine, one request at a time, on the kit's 750-row sample (6,616 questions). The released 0.3.1 changes only the chat route; its decision code is byte-identical to 0.3.0. For E4B and E2B, 0.3.1 itself was also run on the same sample, on one RTX A5000.

| Size | Same answer, 0.3.0 | Per question, largest probability change (median / max) | Same answer, 0.3.1 |
|---|---:|---:|---:|
| 31B | 6,602 (99.79%) | 2.7e-5 / 0.085 | |
| 12B | 6,597 (99.71%) | 3.2e-5 / 0.156 | |
| E4B | 6,604 (99.82%) | 3.2e-5 / 0.149 | 6,605 (99.83%) |
| E2B | 6,600 (99.76%) | 4.3e-5 / 0.089 | 6,602 (99.79%) |

"Same answer" is the same option, or for yes/no questions the same side of 0.5.

- **Tokenizer revisions.** Later revisions of the 12B, E4B and E2B base models change only `tokenizer.json`. Under the latest revision as of 2026-11-09, all 140,620 public requests get the same token ids as under the revisions above.
- **Reproduction script.** `tools/reproduce_public_run.py` in diorite-serve answers the suite's request files offline the same way as the runner (paths are arguments). It was checked end to end on 31B: 80 requests drawn from the kit's 750-row sample, 622 of 622 submitted decisions reproduced. See `PROVENANCE.md` in each results dataset for the hashes of the runner, the converter and every file.

## Request time

Self-measured, for reference only: the kit's 750-row sample (`decision-index suite sample --n 750`; the sample weights every benchmark equally), one RTX PRO 6000, one request at a time, diorite-serve 0.3.0 with the command above.

| Size | Median | Mean | p80 | p95 |
|---|---:|---:|---:|---:|
| 31B | 150 ms | 519 ms | 1,082 ms | 1,983 ms |
| 12B | 65 ms | 212 ms | 427 ms | 775 ms |
| E4B | 46 ms | 98 ms | 192 ms | 334 ms |
| E2B | 36 ms | 63 ms | 112 ms | 192 ms |

Reweighted to the per-benchmark counts of your latency-v1 sample (`data/methodology.json`), the same rows give the numbers below. The sample's 50 ForecastBench rows are left out, since the 0.3 suite no longer runs that benchmark, so 700 of its 750 timed rows are covered. Home appliance simulator, ContractNLI and SimpleBench, which latency-v1 does not time, get weight 0.

| Size | Median | Mean | p80 | p95 |
|---|---:|---:|---:|---:|
| 31B | 97 ms | 290 ms | 323 ms | 1,309 ms |
| 12B | 48 ms | 119 ms | 130 ms | 509 ms |
| E4B | 45 ms | 60 ms | 66 ms | 212 ms |
| E2B | 36 ms | 40 ms | 41 ms | 121 ms |

## Training data

182,167 text questions, the same for all four sizes. Most come from public datasets, including the source datasets of several Decision Index benchmarks; the rest are programmatically generated or synthetic. The training targets are probability distributions over the options. The training data was decontaminated against the Decision Index public test questions.

## Declared limits

The server refuses prompts of 65,536 tokens or more as unsupported. Nothing is truncated and no option is removed.
