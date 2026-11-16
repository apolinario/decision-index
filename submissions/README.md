# submissions: add Qwen3-0.6B joint-final, Public 0.3 index 24.99


**Public Index: 24.99** — 140,178 / 140,178 scored requests `ok`
(`complete: true`; 140,620 total executed requests).

- **Model:** [engageassume/jev-qwen3-0.6b-joint-final-2499](https://huggingface.co/engageassume/jev-qwen3-0.6b-joint-final-2499/tree/9d4d3c44e8946778a899b7c893cbd36eefd5b4f7) @ `9d4d3c4` — unmerged LoRA classifier .
- **Results & inference code:** [pinned release](https://huggingface.co/datasets/engageassume/jev-qwen3-0.6b-joint-final-2499-results/tree/eb7f98e90a42fd4fdc563de4f1665e822442db50) @ `eb7f98e` ([scores.json](https://huggingface.co/datasets/engageassume/jev-qwen3-0.6b-joint-final-2499-results/blob/eb7f98e90a42fd4fdc563de4f1665e822442db50/results/scores.json)).
- **Engine:** `fast_pair_engine:FastPairClassifierEngine`; Transformers SDPA with prefix forks and CUDA Graph replay.
- **Settings:** `max_microbatch_pairs=32`, `max_microbatch_tokens=32768`, `graph_reserved_gib=32`; BF16 base, saved FP32 adapter/head.
- **Kit:** [25c6ec7](https://github.com/apolinario/decision-index/tree/25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d), public edition `0.3`.
- **Hardware:** 1 × NVIDIA H800 NVL, single process.
- **Limits:** 40,960 tokens per candidate
- **Reproduce:** [dependencies and exact launch command](https://huggingface.co/datasets/engageassume/jev-qwen3-0.6b-joint-final-2499-results/blob/eb7f98e90a42fd4fdc563de4f1665e822442db50/README.md).


