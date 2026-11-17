# Submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Inference settings | Declared limits |
|---|---|---:|---|---|---|---|---|
| Adjudex-4B v6 ([`newsbang/adjudex-4b-v6`](https://huggingface.co/newsbang/adjudex-4b-v6/tree/55b02343a204026b3cfed0a30f9ed0f676f38155) @ `55b0234`) | 0.3 | 67.95 | [`runs/adjudex-4b-v6`](https://huggingface.co/datasets/newsbang/decision-index-results/tree/f9f80c669e45c35a11eef54f9263a0bb7c238b0d/runs/adjudex-4b-v6) | kit engine `http` against the model repo's `/v1/systemone` server (`source/`, `python -m autojev.server`, the Perplexity Decider v1.1 server unchanged), model `jev-latest`; kit `25c6ec7`, unmodified runner and scorer | NVIDIA A800 80GB | BF16; one forward pass per question, nothing generated; calibration temperature from `decision_config.json`; default server settings | question branches over 8,192 tokens are rejected (HTTP 422, no truncation): 345 requests, 30 of them scored |
