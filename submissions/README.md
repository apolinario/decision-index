# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [KnowLine-26B-A4B-Gen2.5](https://huggingface.co/PelaAI/KnowLine-26B-A4B-Gen2.5) | 0.3 | 68.21 | [scores.json](https://huggingface.co/datasets/PelaAI/KnowLine-26B-A4B-Gen2.5-decision-index/blob/d1a93be22d97788a32d3e2c29feb53eabac5e7f8/runs/KnowLine-26B-A4B-Gen2.5/scores.json) | `http` against `knowline_server.py` + `knowline_layout.py` from [PelaAI/KnowLine-26B-A4B-Gen2.5](https://huggingface.co/PelaAI/KnowLine-26B-A4B-Gen2.5/tree/7486767f8546beb510e61aef07d0e74219ceb4e3) @ `7486767`, behind SGLang 0.5.21 (bf16), `chat` style, temperature 1, read twice up to 3,000 tokens; runner and scorer `62d2f51` | 4× NVIDIA H20 96 GB, one server per GPU; 16 client shards in parallel per server | 64 questions per request, 255 options per question; no truncation; 0 unsupported; 1 failed request (not in the index) |
