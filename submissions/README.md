# Submissions

| Model | Edition | Public index | Results | Engine and settings | Hardware | Limits |
|---|---|---|---|---|---|---|
| [KnowLine-9B-Gen2](https://huggingface.co/PelaAI/KnowLine-9B-Gen2/tree/8ec8eb609bc10aa6050ae3754aee419d458fae42) | 0.3 | 69.00 | [scores.json](https://huggingface.co/datasets/PelaAI/KnowLine-9B-Gen2-decision-index/blob/1b6bf62bd31427936221f374cff35d3d23145cab/runs/KnowLine-9B-Gen2/scores.json) | `http` against `knowline_server.py` @ `8ec8eb609bc10aa6050ae3754aee419d458fae42`; official kit `62d2f51de34a2de64906345b6bc3e98e27ff55c7`; SGLang 0.5.21 bf16, read twice up to 3000 tokens, folded T=1.63, score questions / 0.6135 | 2 × NVIDIA H20 96 GB; 16 round-robin shards | 64 questions, 255 options; no truncation; 0 unsupported |
