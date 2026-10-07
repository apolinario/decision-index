# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [KnowLine-4B-Gen2](https://huggingface.co/PelaAI/KnowLine-4B-Gen2) | 0.3 | 62.54 | [scores.json](https://huggingface.co/datasets/PelaAI/KnowLine-4B-Gen2-decision-index/blob/26d2c2460a46680e9eebefbd94b18b65ffab16ee/runs/KnowLine-4B-Gen2/scores.json) | `http` against `knowline_server.py` from [PelaAI/KnowLine-4B-Gen2](https://huggingface.co/PelaAI/KnowLine-4B-Gen2/tree/2360ac16a63ab805bb101fcd76a6c58ff287da45) @ `2360ac1`, behind SGLang 0.5.21 (FP8 at load), `chat` style, temperature 1; runner and scorer `62d2f51` | NVIDIA H20 96 GB, one server; 16 client shards in parallel | 64 questions per request, 255 options per question; no truncation; 0 unsupported |
