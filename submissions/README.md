# Submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Inference settings | Declared limits |
|---|---|---:|---|---|---|---|---|
| Adjudex-9B ([`newsbang/adjudex-9b`](https://huggingface.co/newsbang/adjudex-9b/tree/9dca38b6259c21fa698cb21168ee2d0b38a78c8a) @ `9dca38b`) | 0.3 | 62.51 | [`runs/adjudex-9b`](https://huggingface.co/datasets/newsbang/decision-index-results/tree/03a21f3d8f293637a697a38e94cb0087cee4025f/runs/adjudex-9b) | kit engine `http` against the model repo's `/v1/systemone` server (`source/`, `python -m autojev.server`, the Perplexity Decider v1.1 server unchanged), model `jev-latest`; kit `25c6ec7`, unmodified runner and scorer | NVIDIA A800 80GB | BF16; one forward pass per question, nothing generated; calibration temperature from `decision_config.json`; default server settings | question branches over 8,192 tokens are rejected (HTTP 422, no truncation): 345 requests, 30 of them scored |
