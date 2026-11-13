# Submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Inference settings | Declared limits |
|---|---|---:|---|---|---|---|---|
| Adjudex-27B v5 ([`newsbang/adjudex-27b-v5`](https://huggingface.co/newsbang/adjudex-27b-v5/tree/40de55289a10685189e401d0607da15c0b3e9931) @ `40de552`) | 0.3 | 72.41 | [`runs/adjudex-27b-v5`](https://huggingface.co/datasets/newsbang/decision-index-results/tree/93bbb0a61b266363fdcc36f3fb786245eee338f4/runs/adjudex-27b-v5) | kit engine `http` against the model repo's `/v1/systemone` server (`source/`, `python -m autojev.server`, the Perplexity Decider v1.1 server unchanged), model `jev-latest`; kit `25c6ec7`, unmodified runner and scorer | NVIDIA A800 80GB | BF16; one forward pass per question, nothing generated; calibration temperature from `decision_config.json`; default server settings | question branches over 8,192 tokens are rejected (HTTP 422, no truncation): 345 requests, 30 of them scored |
