# Decision Index Submissions

This directory tracks community and author submissions to the [Decision Index](https://github.com/apolinario/decision-index) benchmark.

## Submission Requirements
As described in [README.md](../README.md#submitting-a-model-to-the-leaderboard):
1. **Full 0.2 Suite**: Run all 40 index benchmarks across the 5 evaluation areas without truncation or option filtering.
2. **Complete & Untouched**: All 151,034 requests must be answered (`scores.json` contains `"complete": true`).
3. **Public Results Dataset**: Upload the run directory containing `scores.json`, `index.json`, `benchmark-summary.json`, `environment.json`, and `results.jsonl.gz` to a public Hugging Face dataset repo.
4. **Engine & Hardware Provenance**: Include the exact engine implementation, commit hash, and hardware specifications used for inference.

---

## Submissions Table

| Model | Edition | Decision Index (0.2) | Raw Index | Hardware | Engine / Commit | Results Link | Notes |
|---|---|---:|---:|---|---|---|---|
| **Gevva e2b** | 0.2 | **26.79** | 44.83 | 1x NVIDIA RTX 5090 | `gevva` (`add-gevva-engine`) | [runs/gevva-e2b-0.2](https://huggingface.co/datasets/davidburhans/decision-index-results/tree/main/runs/gevva-e2b-0.2) | 100% complete (151,034 requests). Non-autoregressive System 1 decision engine based on Gemma 4 E2B-it. |
| **Gevva e4b** | 0.2 | **29.88** | 47.50 | 1x NVIDIA RTX 5090 | `gevva` (`add-gevva-engine`) | [runs/gevva-e4b-0.2](https://huggingface.co/datasets/davidburhans/decision-index-results/tree/main/runs/gevva-e4b-0.2) | 100% complete (151,034 requests). Flagship 4.5B model with Shared Prefix KV Cache acceleration. |
