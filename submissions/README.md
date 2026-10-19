# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [KnowLine-26B-A4B-Gen2](https://huggingface.co/PelaAI/KnowLine-26B-A4B-Gen2) | 0.3 | 66.94 | [scores.json](https://huggingface.co/datasets/PelaAI/KnowLine-26B-A4B-Gen2-decision-index/blob/06a5af384e6f50a46acb07c5707c4ff970788535/runs/KnowLine-26B-A4B-Gen2/scores.json) | `http` against `knowline_server.py` from [PelaAI/KnowLine-26B-A4B-Gen2](https://huggingface.co/PelaAI/KnowLine-26B-A4B-Gen2/tree/284787c8b35c7d01da431be44ec3782db16a0704) @ `284787c`, behind SGLang 0.5.21 (bf16), `chat` style, temperature 1; runner and scorer `62d2f51` | 6× NVIDIA H20 96 GB, one server per GPU; 10 or 11 client shards in parallel per server | 64 questions per request, 255 options per question; no truncation; 0 unsupported |
