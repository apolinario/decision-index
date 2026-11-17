# Qwen3.5-9B Base

The unmodified public [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) base model, pinned at revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
No fine-tuning, adapter, merge, prompt search or answer post-processing was used.

| | |
|---|---|
| Decision Index 0.3 public index | **4.69** (raw 24.12, breadth 4.45) |
| Areas | knowledge 0.62 · language 6.11 · retrieval 4.33 · tools 8.17 · arts 1.00 |
| Run | complete: 140,178 of 140,178 scoreable requests; 140,172 `ok`, 6 scoreable `unsupported`, 0 errors |
| Edition rows | 140,620 total rows; 442 excluded by the frozen edition; 321 requests refused by the 8,192-token context limit |
| Request time in `scores.json` | median 50.5 ms, mean 166.3 ms, p95 425.2 ms |

## Run details

- Engine: the official `transformers` engine from Decision Index kit commit `25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d`.
- Model: `Qwen/Qwen3.5-9B` at revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- Settings: `dtype=bfloat16`, `attn=sdpa`, `max_tokens=8192`, `cache_prefix=true`, `chat_template=true`; option scores use the official option-key scoring path.
- Hardware: 8 NVIDIA H20 GPUs, one deterministic `run_id` shard per process. The displayed request times are the sequential per-request times recorded by the kit runner on each shard.
- Results: [complete run directory](https://huggingface.co/datasets/da123/decision-index-results/tree/f47c2a1974ed6f4cecbd2ebcb986788424ff7398/runs/qwen35-9b-public-0.3-transformers-20261117), including `scores.json`, `environment.json`, `status.json`, `benchmark-summary.json`, `index.json` and `results.jsonl.gz`.

## Declared limits

The engine refuses prompts whose prompt plus option exceeds 8,192 tokens. It does not truncate those prompts. The run contains 321 such unsupported rows and no runtime errors.

The public run uses the base model without a custom decision-model inference server. The maintainers may rerun or rescore it with the submitted model and the official kit.
