# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [KnowLine-4B-Gen3](https://huggingface.co/PelaAI/KnowLine-4B-Gen3) | 0.3 | 63.11 | [scores.json](https://huggingface.co/datasets/PelaAI/KnowLine-4B-Gen3-decision-index/blob/0b35c4585375264e2fd238fe1159b86b6bf7301f/runs/KnowLine-4B-Gen3/scores.json) | `http` against `knowline_server.py` from [PelaAI/KnowLine-4B-Gen3](https://huggingface.co/PelaAI/KnowLine-4B-Gen3/tree/d0dad8993aca63bf610c17fd9f71d02981e7e7f3) @ `d0dad89`, behind SGLang 0.5.21 (FP8 at load), `chat` style, temperature 1; runner and scorer `62d2f51` | 4× NVIDIA H20 96 GB, one server per GPU; 16 client shards in parallel per server | 64 questions per request, 255 options per question; no truncation; 0 unsupported |
