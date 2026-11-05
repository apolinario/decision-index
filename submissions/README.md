# Submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Inference settings | Declared limits |
|---|---|---:|---|---|---|---|---|
| Adjudex-9B v3 ([`newsbang/adjudex-9b-v3`](https://huggingface.co/newsbang/adjudex-9b-v3/tree/8669d4db8484dacf50eb5adbe9684d66c1bd5195) @ `8669d4d`) | 0.3 | 67.44 | [`runs/adjudex-9b-v3`](https://huggingface.co/datasets/newsbang/decision-index-results/tree/a74ec54b90e458597d183fee01319041a76e7a60/runs/adjudex-9b-v3) | kit engine `http` against the model repo's `/v1/systemone` server (`source/`, `python -m autojev.server`, the Perplexity Decider v1.1 server unchanged), model `jev-latest`; kit `25c6ec7`, unmodified runner and scorer | NVIDIA A800 80GB | BF16; one forward pass per question, nothing generated; calibration temperature from `decision_config.json`; default server settings | question branches over 8,192 tokens are rejected (HTTP 422, no truncation): 345 requests, 30 of them scored |
