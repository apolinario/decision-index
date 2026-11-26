# Model submissions

| Model | HF weights / revision | Public results | Inference code / commit | Hardware and settings |
|---|---|---|---|---|
| Senova-4B-v5 | [Weights @ 75d0c65](https://huggingface.co/WangSenQWQ/Senova-4B-v5/tree/75d0c655f1a6cfc6185c2aa75840db254f7280ce) (private; contact submitter for review access) | [Public 0.3 index 64.15, scores.json @ 00846a8](https://huggingface.co/datasets/WangSenQWQ/Senova-4B-v5-decision-index-results/blob/00846a8193661564d0acd8c795a996cfca54e22d/runs/senova_4b_v5_readtwice/scores.json) | [Code @ b76bd59](https://github.com/wswsddd/decision-index/tree/b76bd5917516a78b308edbd91dcd217b0fcddc7f) | Public run: 4 nodes × 8 H100 GPUs, one BF16 replica per GPU, SGLang 0.5.15, TP=1 / DP=8, 24 clients, static memory fraction 0.85, 64 running requests per endpoint. Read Twice on, thinking off, unified-letter labels, native selected-token logprobs. Reviewer setup: one GPU, BF16, TP=1 / DP=1, 16 MiB HTTP body limit. |
