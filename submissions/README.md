# Model submissions

| Model | Model revision | Public results | Engine | Hardware and settings | Capacity / notes |
|---|---|---|---|---|---|
| IF-decision-27b v1 | [eea6f72d](https://huggingface.co/peng12345/IF-decision-27b/tree/eea6f72d588183d53fdf6cd312cf356e72e7f9eb) | [Text Public 0.3: 70.92, raw 77.51, coverage 100%](https://huggingface.co/datasets/peng12345/decision-index-results/blob/81cb8444d7480f47b4ec7e6a9f71b9e1da419d3c/runs/IF-decision-27b-v1-text-public-0.3-28ef925de8/scores.json) | [Custom `/v1/systemone` server at e559856](https://github.com/wuyu112/IF-decision-inference/tree/e5598568c8aace2c59088a29c5a8318743820ccf) | A100-SXM4-80GB; bfloat16; full-attention prompt layers bidirectional; vLLM 0.31.0 | 26 options per pass; published wide-choice method scores every candidate and returns every original key above 26. Public run: 140,620 `ok`, coverage 100%. |
