# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [KnowLine-35B-A3B-Gen1.5](https://huggingface.co/PelaAI/KnowLine-35B-A3B-Gen1.5) | 0.3 | 70.72 | [scores.json](https://huggingface.co/datasets/PelaAI/KnowLine-35B-A3B-Gen1.5-decision-index/blob/3bb1747f1ede130b38fa04a2a20a944e6ddb91af/runs/KnowLine-35B-A3B-Gen1.5/scores.json) | `http` against `knowline_server.py` + `knowline_layout.py` from [PelaAI/KnowLine-35B-A3B-Gen1.5](https://huggingface.co/PelaAI/KnowLine-35B-A3B-Gen1.5/tree/9adcf717361b2f7d2946d093f1e85efdf1e59010) @ `9adcf71`, behind SGLang 0.5.21 (FP8 at load), `chat` style, temperature 1, read twice up to 3,000 tokens; runner and scorer `62d2f51` | 4× NVIDIA H20 96 GB, one server per GPU; 16 client shards in parallel per server | 64 questions per request, 255 options per question; no truncation; 0 unsupported |
