# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [KnowLine-4B-Gen4](https://huggingface.co/PelaAI/KnowLine-4B-Gen4) | 0.3 | 64.90 | [scores.json](https://huggingface.co/datasets/PelaAI/KnowLine-4B-Gen4-decision-index/blob/9baa5920fd0898f3ea26e3eece158095063dd4b4/runs/KnowLine-4B-Gen4/scores.json) | `http` against `knowline_server.py` from [PelaAI/KnowLine-4B-Gen4](https://huggingface.co/PelaAI/KnowLine-4B-Gen4/tree/5482f964ba221802a3080d6a73d9352ca85cd1e3) @ `5482f96`, behind SGLang 0.5.21 (FP8 at load), `chat` style, temperature 1; runner and scorer `62d2f51` | 4× NVIDIA H20 96 GB, one server per GPU; 16 client shards in parallel per server | 64 questions per request, 255 options per question; no truncation; 0 unsupported |
