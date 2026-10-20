# Submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| Trio-Spark DI ([`MachineFi/Trio-Spark-DI`](https://huggingface.co/MachineFi/Trio-Spark-DI) @ `e83ab25`) | 0.3 | 55.50 | [`runs/trio-spark-di`](https://huggingface.co/datasets/MachineFi/Trio-Spark-DI-decision-index/tree/af08d192dbd6c7373b89475093c78e06d6440fdf/runs/trio-spark-di) | `http` engine against the repo's `/v1/systemone` server (`serve/`: llama.cpp b11249 + `determinism_server.py`), kit `f5a6e4f` | RTX PRO 6000 96GB, 10 GPUs in parallel, 6 single-slot servers per GPU under CUDA MPS, one request at a time per server | 8,192 tokens per decision prompt; longer requests refused, never truncated; more than 8 options decided by an 8-way tournament, no options removed |
