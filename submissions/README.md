# Model submissions

| Model | Model revision | Public results | Engine | Hardware and settings | Capacity / notes |
|---|---|---|---|---|---|
| IF-decision-27b v1 | [49cb2aea](https://huggingface.co/peng12345/IF-decision-27b/tree/49cb2aea2b54cbaabf2344c5939d689d1d5178ce) | [Text Public 0.3: 70.92, raw 77.51, coverage 100%](https://huggingface.co/datasets/peng12345/decision-index-results/blob/89dc8828bbe1e601d7104e58d9197d9a0a88c32d/runs/IF-decision-27b-v1-text-public-0.3-28ef925de8/scores.json) | Custom /v1/systemone server; source access pending | A100-SXM4-80GB; bfloat16; full-attention prompt layers bidirectional; temperature 1.0 | Native 26 options; grouped multi-pass above 26 is pending complete-candidate and cross-group calibration validation. Pre-submission only; do not review yet. |
