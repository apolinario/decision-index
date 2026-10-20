# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [KnowLine-4B-Gen5](https://huggingface.co/PelaAI/KnowLine-4B-Gen5) | 0.3 | 65.67 | [scores.json](https://huggingface.co/datasets/PelaAI/KnowLine-4B-Gen5-decision-index/blob/a3e9dde571328c9c6581e5da1c6fbf2861cb24f4/runs/KnowLine-4B-Gen5/scores.json) | `http` against `knowline_server.py` from [PelaAI/KnowLine-4B-Gen5](https://huggingface.co/PelaAI/KnowLine-4B-Gen5/tree/a8e0cd664f88bd2d2561dc69de74fdc48c641da5) @ `a8e0cd6`, behind SGLang 0.5.21 (FP8 at load), `chat` style, temperature 1; runner and scorer `62d2f51` | 4× NVIDIA H20 96 GB, one server per GPU; 16 client shards in parallel per server | 64 questions per request, 255 options per question; no truncation; 0 unsupported |
