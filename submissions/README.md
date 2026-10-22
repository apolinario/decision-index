# Model submissions

| Model | HF weights / revision | Public results | Inference code / commit | Hardware and settings |
|---|---|---|---|---|
| Senova-4B | [weights @ 58bcb8a](https://huggingface.co/WangSenQWQ/Senova-4B/tree/58bcb8a28cff3453b1fc401ae79beca3742601c8) | [scores.json @ e997458](https://huggingface.co/datasets/WangSenQWQ/Senova-4B-decision-index-results/blob/e9974586ec8b122ce5129bf714f5f8ba3a547bc7/runs/qwen3.5_4b_pointwise_v53_full_ck1047/scores.json) | [code @ 196b02f](https://github.com/wswsddd/inference-code/tree/196b02f8f8c68165e55b921e82e38c986390e160) | Public: NVIDIA H100 80GB, 8 endpoints, 64 workers, TP=2/DP=4. Deployment: one RTX PRO 6000 96 GB, TP=1/DP=1. Exact settings and limits: code README. |
