# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [KnowLine-4B-Gen5.5](https://huggingface.co/PelaAI/KnowLine-4B-Gen5.5) | 0.3 | 66.75 | [scores.json](https://huggingface.co/datasets/PelaAI/KnowLine-4B-Gen5.5-decision-index/blob/6feb2694c7396d62e7237ed127145bce6a0c8a3a/runs/KnowLine-4B-Gen5.5/scores.json) | `http` against `knowline_server.py` + `knowline_layout.py` from [PelaAI/KnowLine-4B-Gen5.5](https://huggingface.co/PelaAI/KnowLine-4B-Gen5.5/tree/779944d2cc8ffa19cef386aee83c83bcb967ade2) @ `779944d`, behind SGLang 0.5.21 (FP8 at load), `chat` style, temperature 1, read twice up to 3,000 tokens; runner and scorer `62d2f51` | 2× NVIDIA H20 96 GB, one server per GPU; 16 client shards in parallel per server | 64 questions per request, 255 options per question; no truncation; 0 unsupported |
