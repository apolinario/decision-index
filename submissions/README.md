# Submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| Trio-Spark DI ([`MachineFi/Trio-Spark-DI`](https://huggingface.co/MachineFi/Trio-Spark-DI) @ `db04c71`) | 0.3 | 56.74 | [`runs/trio-spark-di`](https://huggingface.co/datasets/MachineFi/Trio-Spark-DI-decision-index/tree/322676e168f8bae5a840e0b4957413226554c267/runs/trio-spark-di) | `http` engine against the repo's `/v1/systemone` server (`serve/`: llama.cpp b11249 + `determinism_server.py`, `trio-spark-di-q8_0.gguf`), kit `f5a6e4f` | RTX PRO 6000 96GB, 10 GPUs in parallel, 6 single-slot servers per GPU under CUDA MPS, one request at a time per server | 8,192 tokens per decision prompt; longer requests refused, never truncated; more than 8 options decided by an 8-way tournament, no options removed |
