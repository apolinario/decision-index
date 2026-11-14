# Submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Inference settings | Declared limits |
|---|---|---:|---|---|---|---|---|
| Adjudex-0.8B v6 ([`newsbang/adjudex-0.8b-v6`](https://huggingface.co/newsbang/adjudex-0.8b-v6/tree/5eac838ef8912c59cb36a746606cef834bfe3c3c) @ `5eac838`) | 0.3 | 55.09 | [`runs/adjudex-0.8b-v6`](https://huggingface.co/datasets/newsbang/decision-index-results/tree/7da81b672578171194f9f21ac66372ff0c14a198/runs/adjudex-0.8b-v6) | kit engine `http` against the model repo's `/v1/systemone` server (`source/`, `python -m autojev.server`, the Perplexity Decider v1.1 server unchanged), model `jev-latest`; kit `25c6ec7`, unmodified runner and scorer | NVIDIA A800 80GB | BF16; one forward pass per question, nothing generated; calibration temperature from `decision_config.json`; default server settings | question branches over 8,192 tokens are rejected (HTTP 422, no truncation): 345 requests, 30 of them scored |
