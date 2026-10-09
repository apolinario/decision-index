# Xor 26B-A4B v1.1

Self-run Decision Index 0.3 public score **57.62** (raw 67.88, breadth 56.89). Complete: 140620 requests all ok; 140178 scored after the official 442 exclusions. No errors, unsupported requests or abstentions. This is not the private Full score or a maintainer-verified rank.

- Model: https://huggingface.co/juspay/Xor-26B-A4B/tree/28ca09c7b76648b24d3561c940d1a7460d12fb8e (tag v1.1).
- Results: https://huggingface.co/datasets/juspay/xor-26b-a4b-v1.1-decision-index/tree/170d39fa196ef0f4a5d0ec562a6079024084c08f/runs/xor-26b-a4b-v1.1
- Kit and HTTP engine: apolinario/decision-index @ 9eb2dbe2a358004c8782c66e40a83ac07b953fec.
- Inference wrapper: serving/xn_base_server.py at the model revision; SHA-256 2eed569347cd5c6d4afbc638c0feb1b3cdbbb4dff02ed927788d9ded8cf4e435.
- SGLang image: prakhar1611/xor-sglang@sha256:94c48d2a6cc98dc456cf93f723707ea7dd81dddfe1061e823b348d68bbe8158f. Contains the candidate-logprob tensor/list handling fix.

## Run identity and packaging

Fresh uninterrupted run on six NVIDIA B300 SXM6 AC GPUs (275040 MiB each), one isolated TP1 replica per GPU. Twenty-four unchanged upstream sequential runner processes handled disjoint row shards, four per replica. No prior responses reused; all worker exit codes zero. Results were merged and scored by the unchanged upstream scorer.

Published results are a compact projection of that original run: only payload and raw_output fields were removed, matching upstream compact mode. Answers, statuses, IDs and timings are unchanged. The original full traces are retained privately. Re-scoring the compact results with the pinned kit reproduces 57.62.

The aggregate reports and status are copied unchanged from the original run. environment.json removes three local suite-file paths, records the original environment hash, and retains hardware/settings/capacity. PROVENANCE.json records the original and compact result hashes. Benchmark source texts and raw backend outputs are not redistributed.

Original full results SHA-256: c3402155074b3eb24dcb720d5aa642665408a57a2d1c86c3f88d99bfbb12633f.
Compact uncompressed SHA-256: 325309dd16b88d1e67ec0ff560befbbe8c9877703cf996dcfcc74934989abf01.

Recorded loaded HTTP latency: median 137.0 ms, mean 196.4 ms, p95 461.6 ms. Not a single-GPU sequential eligibility measurement. Please run the independent latency and held-out-answer checks.

## Reproducible serving

Linux NVIDIA host, Docker Compose GPU support and NVIDIA Container Toolkit; tested single-GPU release configuration on RTX PRO 6000 96 GB.

```bash
hf download juspay/Xor-26B-A4B --revision 28ca09c7b76648b24d3561c940d1a7460d12fb8e --local-dir ./xor-v1.1
export MODEL_DIR="$(pwd)/xor-v1.1"
mkdir -p ./xor-v1.1-runtime
tar -xzf "$MODEL_DIR/serving/xor-v1.1-serving.tar.gz" -C ./xor-v1.1-runtime
cd ./xor-v1.1-runtime
export GPU_ID=0
export API_PORT=8000
bash ./run.sh
```

Then, from the pinned kit checkout with a rebuilt, verified suite:

```bash
python -m decision_index pipeline --engine http --edition 0.3 \
  --option base_url=http://127.0.0.1:8000 \
  --option model=xor-26b-a4b-xl5050 \
  --suite-dir /path/to/suite-0.3 --compact --out runs/xor-26b-a4b-v1.1
```

This is the single-client reproduction path, not the original 24-shard topology. The API binds to localhost and has no authentication; do not expose directly.

Inference: merged BF16 weights; TP1; memory fraction 0.8; context and max prefill 131072; chunked prefill 16384. Two-order scoring enabled; temperatures choice=2.0, noul=1.7, score=1.0. Candidate-token logprobs with one output token per scoring row; normalize over all candidates. All question types use one fixed renderer, with no per-benchmark prompt changes.

Capacity: choice 2–255 options; score 2–10 levels; 131072-token context; 8 MiB request body. Oversized inputs rejected, not truncated. Image inputs: 1–8 base64 data URLs; alternatively one video data URL; no remote URL fetching or audio. No scored vision evaluation is claimed.

## Provenance and limitations

Post-trained google/gemma-4-26B-A4B-it, base revision 4d7ae4984b7db7de8f8457170b3f1a419ee76d52, checkpoint xn-lora-xl-5050. Complete adapter blend reconstruction is not established by the available provenance. Training/public-benchmark overlap has not been independently established; no benchmark-clean training claim is made. Exact evaluated merged weights and wrapper are pinned in the release.

## Vision evaluation request

Please also consider this image-capable release for the Vision board. Image/video smoke checks passed; 57.62 is solely the public text-suite score. Could you provide the vision harness and required artifacts, or confirm whether maintainers run both vision portions? Please confirm vision input-format and latency requirements. The same pinned wrapper supports images alongside state and typed questions, as documented in the model card.
