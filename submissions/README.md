# Submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Inference settings | Declared limits |
|---|---|---:|---|---|---|---|---|
| Adjudex-0.8B v7 ([`newsbang/adjudex-0.8b-v7`](https://huggingface.co/newsbang/adjudex-0.8b-v7/tree/3cce6aa8433abf809ec3dcbb760630d9d064a31e) @ `3cce6aa`) | 0.3 | 54.77 | [`runs/adjudex-0.8b-v7`](https://huggingface.co/datasets/newsbang/decision-index-results/tree/c90e1ac40493f0baa19ec58fc663609b814703e5/runs/adjudex-0.8b-v7) | kit engine `http` against the model repo's `/v1/systemone` server (`source/`, `python -m autojev.server`, the Perplexity Decider v1.1 server unchanged), model `jev-latest`; kit `25c6ec7`, unmodified runner and scorer | NVIDIA A800 80GB | BF16; one forward pass per question, nothing generated; calibration temperature from `decision_config.json`; default server settings | question branches over 8,192 tokens are rejected (HTTP 422, no truncation): 345 requests, 30 of them scored |
