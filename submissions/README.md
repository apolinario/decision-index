# Submissions

| Model | Edition | Public index | Results | Engine and settings | Hardware | Limits |
|---|---|---|---|---|---|---|
| [KnowLine-31B-Gen2](https://huggingface.co/PelaAI/KnowLine-31B-Gen2/tree/fd99a9ed2255d8a67dd47a44cc429e638f611593) | 0.3 | 73.00 | [scores.json](https://huggingface.co/datasets/PelaAI/KnowLine-31B-Gen2-decision-index/blob/345db934bdc7f0c6aa22aaa9b030efc0fedbf860/runs/KnowLine-31B-Gen2/scores.json) | `http` against `knowline_server.py` @ `fd99a9ed2255d8a67dd47a44cc429e638f611593`; official kit `62d2f51de34a2de64906345b6bc3e98e27ff55c7`; SGLang 0.5.21 bf16, read twice up to 3000 tokens, folded T=1.27, score questions / 0.7874 | 2 × NVIDIA H20 96 GB; 16 round-robin shards | 64 questions, 255 options; no truncation; 0 unsupported |
