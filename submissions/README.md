# Model submissions

| Model | Model revision | Public results | Engine | Hardware and settings | Capacity / notes |
|---|---|---|---|---|---|
| APUS Reflex2 Decision V1 (DJ600) | [ddfdc6ba](https://huggingface.co/gump2049/APUS-Reflex2/tree/ddfdc6ba307421c96440f37ec31a1d0a04d59468) | [Text Public 0.3: 73.82, raw 80.05, coverage 100%](https://huggingface.co/datasets/gump2049/APUS-Decision-Model-performance/blob/2c9f65df41fcc4340494490d3539a785bbef4eb6/runs/APUS-Reflex-v2-dj-600/scores.json) | APUS/1.0 `/v1/systemone` runtime; public prescreen results used vLLM | H100 SXM 80GB x4 total, two parallel TP=2 services; bfloat16; max model length 131072; max batched tokens 131072; max sequences 8; GPU memory utilization 0.85; eager execution; TRITON_ATTN | Text-only; up to 64 questions, 255 choice options, 10 score levels. Public run: 140,178 scoreable rows `ok`, coverage 100%. Model and runtime are private at submission time. Please grant maintainer access for private/full evaluation. |
