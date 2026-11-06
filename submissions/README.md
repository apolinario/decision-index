# Submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Inference settings | Declared limits |
|---|---|---:|---|---|---|---|---|
| Adjudex-0.8B v4 ([`newsbang/adjudex-0.8b-v4`](https://huggingface.co/newsbang/adjudex-0.8b-v4/tree/717f79498c1e7fb0d1d39273aaba7b6022890353) @ `717f794`) | 0.3 | 55.10 | [`runs/adjudex-0.8b-v4`](https://huggingface.co/datasets/newsbang/decision-index-results/tree/8953bab028f89c6f85fecb085b1d4687c9d94265/runs/adjudex-0.8b-v4) | kit engine `http` against the model repo's `/v1/systemone` server (`source/`, `python -m autojev.server`, the Perplexity Decider v1.1 server unchanged), model `jev-latest`; kit `25c6ec7`, unmodified runner and scorer | NVIDIA A800 80GB | BF16; one forward pass per question, nothing generated; calibration temperature from `decision_config.json`; default server settings | question branches over 8,192 tokens are rejected (HTTP 422, no truncation): 345 requests, 30 of them scored |
