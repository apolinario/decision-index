# Submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Inference settings | Declared limits |
|---|---|---:|---|---|---|---|---|
| Adjudex-4B v7 ([`newsbang/adjudex-4b-v7`](https://huggingface.co/newsbang/adjudex-4b-v7/tree/ff4677a23dff03c4fc21383c77cf5f626ec23108) @ `ff4677a`) | 0.3 | 68.01 | [`runs/adjudex-4b-v7`](https://huggingface.co/datasets/newsbang/decision-index-results/tree/5a50aff0bb8a16d339d99198f3819c9b775a98f9/runs/adjudex-4b-v7) | kit engine `http` against the model repo's `/v1/systemone` server (`source/`, `python -m autojev.server`, the Perplexity Decider v1.1 server unchanged), model `jev-latest`; kit `25c6ec7`, unmodified runner and scorer | NVIDIA A800 80GB | BF16; one forward pass per question, nothing generated; calibration temperature from `decision_config.json`; default server settings | question branches over 8,192 tokens are rejected (HTTP 422, no truncation): 345 requests, 30 of them scored |
