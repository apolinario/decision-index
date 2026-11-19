# Model submissions

| Model | Model revision | Public results | Engine | Hardware and settings | Capacity / notes |
|---|---|---|---|---|---|
| IF-decision-27b v1 | [49cb2aea](https://huggingface.co/peng12345/IF-decision-27b/tree/49cb2aea2b54cbaabf2344c5939d689d1d5178ce) | [Text Public 0.3: 70.92, raw 77.51, coverage 100%](https://huggingface.co/datasets/peng12345/decision-index-results/blob/8490ab8f7e3b18a8cd3efd8d613c0eeadbe1aff8/runs/IF-decision-27b-v1-text-public-0.3-28ef925de8/scores.json) | [Custom `/v1/systemone` server at c26677e](https://github.com/wuyu112/IF-decision-inference/tree/c26677e663f18750478f0f0ccfce91c99dfaaa58) | A100-SXM4-80GB; bfloat16; full-attention prompt layers bidirectional; vLLM 0.31.0 | 26 options per pass; published wide-choice method scores every candidate and returns every original key above 26. Public run: 140,620 `ok`, coverage 100%. |
