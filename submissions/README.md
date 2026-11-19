# Model submissions

| Model | Model revision | Public results | Engine | Hardware and settings | Capacity / notes |
|---|---|---|---|---|---|
| APUS Reflex2 (DJ600) | [c0ac1c8b](https://huggingface.co/gump2049/APUS-Reflex2/tree/c0ac1c8bd43926723f0b029630628b0b695e904b) | [Text public 0.3: 73.82, raw 80.05, coverage 100%](https://huggingface.co/datasets/gump2049/APUS-Reflex2-eval-dataset/blob/0d82f8a104c4b26345c81291017bb224af22c360/runs/APUS-Reflex2-DJ600/scores.json) | APUS/1.0 `POST /v1/systemone`; public run through vLLM 0.33.0 | H100 SXM 80GB x4 total, two TP=2 services; bfloat16; max model length 131072; max batched tokens 131072; max sequences 8; GPU memory utilization 0.85; eager execution; `TRITON_ATTN` | Text-only; up to 64 questions, 255 choice options, 10 score levels. Public run: 140,178 scoreable rows `ok`, coverage 100%. Reference Transformers runtime is included in the public model repository. |
