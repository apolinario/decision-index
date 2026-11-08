# Submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Inference settings | Declared limits |
|---|---|---:|---|---|---|---|---|
| Adjudex-9B v4 ([`newsbang/adjudex-9b-v4`](https://huggingface.co/newsbang/adjudex-9b-v4/tree/cd29307235c7f24002fa75c88ce7340e30fa83b0) @ `cd29307`) | 0.3 | 68.87 | [`runs/adjudex-9b-v4`](https://huggingface.co/datasets/newsbang/decision-index-results/tree/867bdbaf6f23711cc3168d3178de0c75961fb6c9/runs/adjudex-9b-v4) | kit engine `http` against the model repo's `/v1/systemone` server (`source/`, `python -m autojev.server`, the Perplexity Decider v1.1 server unchanged), model `jev-latest`; kit `25c6ec7`, unmodified runner and scorer | NVIDIA A800 80GB | BF16; one forward pass per question, nothing generated; calibration temperature from `decision_config.json`; default server settings | question branches over 8,192 tokens are rejected (HTTP 422, no truncation): 345 requests, 30 of them scored |
