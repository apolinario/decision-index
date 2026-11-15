# Submissions

| Model | Edition | Public index | Results | Engine and settings | Hardware | Limits |
|---|---|---|---|---|---|---|
| [KnowLine-12B-Gen1](https://huggingface.co/PelaAI/KnowLine-12B-Gen1/tree/f26fbc2d64092bb2f8caaedbec243172bb3e908f) | 0.3 | 69.37 | [scores.json](https://huggingface.co/datasets/PelaAI/KnowLine-12B-Gen1-decision-index/blob/131c640c9e7ab8c11fc893e9e5523b37bd8e9dd6/runs/KnowLine-12B-Gen1/scores.json) | `http` against `knowline_server.py` @ `f26fbc2d64092bb2f8caaedbec243172bb3e908f`; official kit `62d2f51de34a2de64906345b6bc3e98e27ff55c7`; SGLang 0.5.21 bf16, read twice up to 3000 tokens, folded T=1.47, score questions / 0.6803 | 2 × NVIDIA H20 96 GB; 16 round-robin shards | 64 questions, 255 options; no truncation; 0 unsupported |
