# Submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Inference settings | Declared limits |
|---|---|---:|---|---|---|---|---|
| Adjudex-9B v2 ([`newsbang/adjudex-9b-v2`](https://huggingface.co/newsbang/adjudex-9b-v2/tree/02816339df094c7cf5db5701f56e8307c8f97754) @ `0281633`) | 0.3 | 63.52 | [`runs/adjudex-9b-v2`](https://huggingface.co/datasets/newsbang/decision-index-results/tree/afa2adfb9ff8a1f8d65275e7bde430aae73d4297/runs/adjudex-9b-v2) | kit engine `http` against the model repo's `/v1/systemone` server (`source/`, `python -m autojev.server`, the Perplexity Decider v1.1 server unchanged), model `jev-latest`; kit `25c6ec7`, unmodified runner and scorer | NVIDIA A800 80GB | BF16; one forward pass per question, nothing generated; calibration temperature from `decision_config.json`; default server settings | question branches over 8,192 tokens are rejected (HTTP 422, no truncation): 345 requests, 30 of them scored |
