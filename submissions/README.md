# Submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Inference settings | Declared limits |
|---|---|---:|---|---|---|---|---|
| Adjudex-9B v5 ([`newsbang/adjudex-9b-v5`](https://huggingface.co/newsbang/adjudex-9b-v5/tree/ffdcb8254d1b22e40b2bbbf569f143e0b357aeca) @ `ffdcb82`) | 0.3 | 69.28 | [`runs/adjudex-9b-v5`](https://huggingface.co/datasets/newsbang/decision-index-results/tree/abaa32c31a79904628276593b0ed34ee507587b5/runs/adjudex-9b-v5) | kit engine `http` against the model repo's `/v1/systemone` server (`source/`, `python -m autojev.server`, the Perplexity Decider v1.1 server unchanged), model `jev-latest`; kit `25c6ec7`, unmodified runner and scorer | NVIDIA A800 80GB | BF16; one forward pass per question, nothing generated; calibration temperature from `decision_config.json`; default server settings | question branches over 8,192 tokens are rejected (HTTP 422, no truncation): 345 requests, 30 of them scored |
