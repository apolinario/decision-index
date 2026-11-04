# Model submissions

| Model | HF weights / revision | Public results | Inference code / commit | Hardware and settings |
|---|---|---|---|---|
| Solara-27B | [weights @ f705198](https://huggingface.co/WangSenQWQ/Solara-27B/tree/f705198fb3719872d7bce3d08dd835bfd99a3998) | [public 0.3 index 72.31; scores.json @ 8b222f9](https://huggingface.co/datasets/WangSenQWQ/Solara-27B-decision-index-results/blob/8b222f95de66ece9ff7e3919f51efa7049125ce2/runs/solara_27b_readtwice/scores.json) | [code @ b432d65](https://github.com/wswsddd/Solara-27B-inference/tree/b432d6525bfabda1457b3f2e1a63be1301aee42d) | Public: 16 nodes × 8 NVIDIA H100 80GB, one BF16 model per GPU. Maintainer: 1× H100 80GB. Read Twice on; thinking off; unified LM-head option softmax. [Exact settings and limits](https://github.com/wswsddd/Solara-27B-inference/blob/b432d6525bfabda1457b3f2e1a63be1301aee42d/README.md#4-exact-inference-settings-and-limits). |
