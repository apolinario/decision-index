# Submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| Trio-Spark DI ([`MachineFi/Trio-Spark-DI`](https://huggingface.co/MachineFi/Trio-Spark-DI) @ `5e54f67`) | 0.3 | 55.52 | [`runs/trio-spark-di`](https://huggingface.co/datasets/MachineFi/Trio-Spark-DI-decision-index/tree/843f97a348159d2bb41b810639896e8785992dd6/runs/trio-spark-di) | `http` engine against the repo's `/v1/systemone` server (`serve/`: llama.cpp b11249 + `spark_server.py`), kit `f5a6e4f` | RTX PRO 6000 96GB, several GPUs in parallel, one server per GPU (16 llama.cpp slots), 16 concurrent requests per GPU | 8,192 tokens per decision prompt; longer requests refused, never truncated; more than 8 options decided by an 8-way tournament, no options removed |
