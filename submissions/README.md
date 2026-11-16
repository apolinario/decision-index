# Submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Inference settings | Declared limits |
|---|---|---:|---|---|---|---|---|
| Adjudex-4B v5 ([`newsbang/adjudex-4b-v5`](https://huggingface.co/newsbang/adjudex-4b-v5/tree/89b79c7a3c6fdb0a6c767323b8cde555b1a7958d) @ `89b79c7`) | 0.3 | 67.41 | [`runs/adjudex-4b-v5`](https://huggingface.co/datasets/newsbang/decision-index-results/tree/3e25e6f2c8a14e2274497b86f2baf8fb0a75abeb/runs/adjudex-4b-v5) | kit engine `http` against the model repo's `/v1/systemone` server (`source/`, `python -m autojev.server`, the Perplexity Decider v1.1 server unchanged), model `jev-latest`; kit `25c6ec7`, unmodified runner and scorer | NVIDIA A800 80GB | BF16; one forward pass per question, nothing generated; calibration temperature from `decision_config.json`; default server settings | question branches over 8,192 tokens are rejected (HTTP 422, no truncation): 345 requests, 30 of them scored |
