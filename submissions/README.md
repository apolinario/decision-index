# Submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Inference settings | Declared limits |
|---|---|---:|---|---|---|---|---|
| Adjudex-0.8B v5 ([`newsbang/adjudex-0.8b-v5`](https://huggingface.co/newsbang/adjudex-0.8b-v5/tree/53e4e12b852138998671c32525f055858e2ad160) @ `53e4e12`) | 0.3 | 54.69 | [`runs/adjudex-0.8b-v5`](https://huggingface.co/datasets/newsbang/decision-index-results/tree/a33c9b617a93ecc791cf9f7ea351ed1b83c04784/runs/adjudex-0.8b-v5) | kit engine `http` against the model repo's `/v1/systemone` server (`source/`, `python -m autojev.server`, the Perplexity Decider v1.1 server unchanged), model `jev-latest`; kit `25c6ec7`, unmodified runner and scorer | NVIDIA A800 80GB | BF16; one forward pass per question, nothing generated; calibration temperature from `decision_config.json`; default server settings | question branches over 8,192 tokens are rejected (HTTP 422, no truncation): 345 requests, 30 of them scored |
