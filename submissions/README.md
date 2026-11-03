# Submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Inference settings | Declared limits |
|---|---|---:|---|---|---|---|---|
| Adjudex-0.8B ([`newsbang/adjudex-0.8b`](https://huggingface.co/newsbang/adjudex-0.8b/tree/8fb432291598e1d02b6f59b8cd723c70fbcaf50f) @ `8fb4322`) | 0.3 | 42.28 | [`runs/adjudex-0.8b`](https://huggingface.co/datasets/newsbang/decision-index-results/tree/abf57c57dac54b430739d0b7fe013328dd5d7cab/runs/adjudex-0.8b) | kit engine `http` against the model repo's `/v1/systemone` server (`source/`, `python -m autojev.server`, the Perplexity Decider v1.1 server unchanged), model `jev-latest`; kit `25c6ec7`, unmodified runner and scorer | NVIDIA A800 80GB | BF16; one forward pass per question, nothing generated; calibration temperature from `decision_config.json`; default server settings | question branches over 8,192 tokens are rejected (HTTP 422, no truncation): 345 requests, 30 of them scored |
