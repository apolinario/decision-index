# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [Decision 4B v1.5](https://huggingface.co/flymy-ai/decision-4b-v1.5/tree/a80e804fdc1545f671d494c411bc025adc807bff) (FlyMy.AI, `decision-index` branch) | 0.3 | 38.24 | [scores.json](https://huggingface.co/datasets/flymy-ai/decision-index-results/blob/5ee0db3c544652d10f23385e81c4f9175f34d675/runs/decision-4b-v1.5-0.3/scores.json) | `http` → the branch `server.py`; kit `9eb2dbe` | 1× NVIDIA RTX 4090 24 GB, bf16, one request at a time | input cap 131,072 tokens, at most 676 options per choice; no truncation, 0 unsupported |
