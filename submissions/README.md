# Submissions

| Model | Public 0.3 Index | Model and engine revision | Results | Hardware | Inference settings and capacity |
| --- | ---: | --- | --- | --- | --- |
| FlyAI-Decision-27B | 65.74 (raw 74.02) | [Weights and engine](https://huggingface.co/FlyaiLab/FlyAI-Decision-27B/tree/6c4c2ba621f4f42d8807e50a682824d397e85e71); engine in `inference/`, revision `6c4c2ba621f4f42d8807e50a682824d397e85e71` | [Complete scores](https://huggingface.co/datasets/FlyaiLab/FlyAI-Decision-27B-results/blob/73bcfaa674b54d3d5e43e511dbe0ddc3cc981b58/runs/FlyAI-Decision-27B-vllm031-v7/scores.json); results revision `73bcfaa674b54d3d5e43e511dbe0ddc3cc981b58` | 8 independent NVIDIA L20Z workers; 1 GPU per process, TP=1, sequential requests | BF16; eager synchronous execution; context 32,768 tokens per field; up to 255 candidates per field; max_num_seqs=64 field prompts; max_num_batched_tokens=32,768; gpu_memory_utilization=0.80; seed=20261023; temperature=0; max_tokens=1; max_logprobs=255; logprobs_mode=processed_logits; prefix caching, chunked prefill and async scheduling disabled; one-token readout with FP32 candidate softmax; 16 bidirectional upstream full-attention layers and 48 unchanged causal Gated DeltaNet layers. |

The complete run covers 140,178 requests and 312,320 fields, with zero errors or
unsupported requests. The official scorer reports `complete: true`. The public
run has median/mean/p80 request latency of 78.06/163.16/144.98 ms on L20Z; official
RTX PRO 6000 latency and the private parts remain for maintainer evaluation.

From the downloaded model repository at the pinned revision:

```bash
export CUDA_VISIBLE_DEVICES=0
bash inference/install.sh
bash inference/run_rows.sh "$PWD" /path/to/rows.jsonl /path/to/output
```

The frozen uv lock pins Python 3.12.12, vLLM 0.31.0, PyTorch 2.13.0/CUDA 13.0,
Transformers 5.17.0 and Tokenizers 0.23.3. The official kit is vendored unchanged
at `25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d`. Installation prerequisites and the
engine source manifest are included in `inference/README.md`.

Over-capacity inputs are rejected without truncation. Results preserve the
original official runner bytes; the dataset includes the merged results, hashes,
coverage audit, dependency lock and runtime receipts. No rebuilt benchmark
inputs or training examples are redistributed.

For a rebuilt public suite, run `bash inference/run_suite.sh "$PWD" /path/to/suite-0.3 /path/to/fresh-output`. This uses the official pipeline with suite hash verification and edition filtering.

The original evaluated revision is `801eeb93168bb28a20a8e6f24d0e7527a2bec3a0`. The reproduction revision in the table changes documentation, startup wrappers and host/compiler preflight only; weights, inference code, prompts, codebook and dependency lock are unchanged. The original results revision is retained. [Maintenance verification](https://huggingface.co/datasets/FlyaiLab/FlyAI-Decision-27B-results/blob/4f620b06079dc57c304963bd65764ddce1c98ec2/reproduction-update-20261124.json) records the earlier startup checks; subsequent RTX validation is recorded below.

## RTX PRO 6000 validation

RTX PRO 6000 reproduction validation completed on 2026-11-24 using the unchanged published revision and its `inference/install.sh` + `inference/run_rows.sh`: 1,330/1,330 requests and 7,691 fields succeeded, including 10 long-input requests (maximum 22,695 input tokens). Host: Ubuntu 24.04, glibc 2.39, GCC 13.3.0, NVIDIA driver 595.112.07; the frozen dependencies and inference settings are unchanged.

On the same preselected 512-request diagnostic sample, median/mean/p80 was 77.85/235.33/214.64 ms (p95 933.09 ms; maximum 6,572.10 ms). Across all 1,330 validation requests, median/mean/p80 was 88.25/497.84/446.38 ms. Percentiles follow the official public scorer's nearest-rank definition. This is submitter-run validation, not the maintainer's private held-out sample or official latency certification.

Decision agreement was 7,682/7,691 with prior L20Z (99.8830%; 9 differences) and 6,424/6,434 with the native reference (99.8446%; 10 differences). All paired inputs were verified, including rehashing all 809 native payloads using their original JSON serialization. Generic single/mixed/reverse-order checks changed zero decisions. [Detailed RTX validation record](https://huggingface.co/datasets/FlyaiLab/FlyAI-Decision-27B-results/blob/caf8c46903fbbcdb50dd9a1afa3be9df93d200b7/rtx-pro-6000-validation-20261124.json). The complete public run, score 65.74, and model revision remain unchanged.
