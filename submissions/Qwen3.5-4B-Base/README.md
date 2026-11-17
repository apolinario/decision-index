# Qwen3.5-4B Base

The unmodified public [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) base model, pinned at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
No fine-tuning, adapter, merge, prompt search or answer post-processing was used.

| | |
|---|---|
| Decision Index 0.3 public index | **4.03** (raw 26.41, breadth 3.88) |
| Areas | knowledge 2.74 · language 4.59 · retrieval 2.38 · tools 8.17 · arts 1.61 |
| Run | complete: 140,178 of 140,178 scoreable requests; 140,172 `ok`, 6 scoreable `unsupported`, 0 errors |
| Edition rows | 140,620 total rows; 442 excluded by the frozen edition; 321 requests refused by the 8,192-token context limit |
| Request time in `scores.json` | median 48.8 ms, mean 136.5 ms, p95 418.9 ms |

## Run details

- Engine: the official `transformers` engine from Decision Index kit commit `25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d`.
- Model: `Qwen/Qwen3.5-4B` at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Settings: `dtype=bfloat16`, `attn=sdpa`, `max_tokens=8192`, `cache_prefix=true`, `chat_template=true`; option scores use the official option-key scoring path.
- Hardware: 8 NVIDIA H20 GPUs, one deterministic `run_id` shard per process. The displayed request times are the sequential per-request times recorded by the kit runner on each shard.
- Results: [complete run directory](https://huggingface.co/datasets/da123/decision-index-results/tree/d9ce3c2b91a12c0345d545486fd7a0295f22e701/runs/qwen35-4b-public-0.3-transformers-20261117), including `scores.json`, `environment.json`, `status.json`, `benchmark-summary.json`, `index.json` and `results.jsonl.gz`.

## Declared limits

The engine refuses prompts whose prompt plus option exceeds 8,192 tokens. It does not truncate those prompts. The run contains 321 such unsupported rows and no runtime errors.

The public run uses the base model without a custom decision-model inference server. The maintainers may rerun or rescore it with the submitted model and the official kit.
