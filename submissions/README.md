# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [KnowLine-26B-A4B-Gen1](https://huggingface.co/PelaAI/KnowLine-26B-A4B-Gen1) | 0.3 | 65.76 | [scores.json](https://huggingface.co/datasets/PelaAI/KnowLine-26B-A4B-Gen1-decision-index/blob/5d20d510a9ae74c3afb55e61fa744af07d558b53/runs/KnowLine-26B-A4B-Gen1/scores.json) | `http` against `knowline_server.py` from [PelaAI/KnowLine-26B-A4B-Gen1](https://huggingface.co/PelaAI/KnowLine-26B-A4B-Gen1/tree/043e8e45ba9ee5231ea73f965b93cdba841d5686) @ `043e8e4`, behind SGLang 0.5.21 (bf16), `chat` style, temperature 1; runner and scorer `62d2f51` | 6× NVIDIA H20 96 GB, one server per GPU; 10 or 11 client shards in parallel per server | 64 questions per request, 255 options per question; no truncation; 0 unsupported |
