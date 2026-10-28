# Submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| Trio-Spark DI 1.4 ([`MachineFi/Trio-Spark-DI-1.4`](https://huggingface.co/MachineFi/Trio-Spark-DI-1.4) @ `fe348f8`) | 0.3 | 56.41 | [`runs/trio-spark-di-1.4`](https://huggingface.co/datasets/MachineFi/Trio-Spark-DI-decision-index/tree/0f78652f2e91f2274251d6480cdcc62af09d059a/runs/trio-spark-di-1.4) | `http` engine against the repo's `/v1/systemone` server (`serve/`: llama.cpp b11249 + `determinism_server.py`, `trio-spark-di-1.4-q8_0.gguf`), kit `25c6ec7` | RTX PRO 6000 96GB, 10 GPUs in parallel, 4 servers per GPU, one request at a time per server | 8,192 tokens per decision prompt; longer requests refused, never truncated; more than 8 options decided by an 8-way tournament, no options removed |
