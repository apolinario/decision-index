# Model submissions

| Model | HF weights / revision | Public results | Inference code / commit | Hardware and settings |
|---|---|---|---|---|
| Senova-4B-v4 | [weights @ 0551d9a](https://huggingface.co/WangSenQWQ/Senova-4B-v4/tree/0551d9add252dca1df7795207ffe765aeb8c55b3) | [public 0.3 index 67.02; scores.json @ 3022a2f](https://huggingface.co/datasets/WangSenQWQ/Senova-4B-v4-decision-index-results/blob/3022a2ff5c4e4214f3ee4615110cf90bc8bf8479/runs/senova_4b_v4_readtwice/scores.json) | [code @ 3bda7ea](https://github.com/wswsddd/Senova-4B-v4-inference/tree/3bda7ea3f5994bef10c717dd505d0991db00a128) | Public: 16 nodes × 8 NVIDIA H100 80GB, one BF16 model per GPU. Maintainer: 1× RTX PRO 6000 96GB. Read Twice on; thinking off; unified LM-head option softmax. [Exact settings and limits](https://github.com/wswsddd/Senova-4B-v4-inference/blob/3bda7ea3f5994bef10c717dd505d0991db00a128/README.md#4-exact-inference-settings-and-limits). |
