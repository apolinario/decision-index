# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [KnowLine-35B-A3B-Gen1](https://huggingface.co/PelaAI/KnowLine-35B-A3B-Gen1) | 0.3 | 69.49 | [scores.json](https://huggingface.co/datasets/PelaAI/KnowLine-35B-A3B-Gen1-decision-index/blob/a33ce28a6c3804cf73be3eda1c10098543635232/runs/KnowLine-35B-A3B-Gen1/scores.json) | `http` against `knowline_server.py` from [PelaAI/KnowLine-35B-A3B-Gen1](https://huggingface.co/PelaAI/KnowLine-35B-A3B-Gen1/tree/e800968d38761c190476da5c69c6a5087a3c9b29) @ `e800968`, behind SGLang 0.5.21 (FP8 at load), `chat` style, temperature 1; runner and scorer `62d2f51` | 4× NVIDIA H20 96 GB, one server per GPU; 16 client shards in parallel per server | 64 questions per request, 255 options per question; no truncation; 0 unsupported |
