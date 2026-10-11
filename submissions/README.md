# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [Decision 4B v1.6](https://huggingface.co/flymy-ai/decision-4b-v1.6/tree/3399e91d615950ff53e08b41fc2ede170503f9a6) (FlyMy.AI; supersedes v1.5) | 0.3 | 42.69 | [scores.json](https://huggingface.co/datasets/flymy-ai/decision-index-results/blob/1743d914d31e2e05d3fb162a280dd3069ca60d11/runs/decision-4b-v1.6-0.3/scores.json) | `http` → the package `server.py`; kit `9eb2dbe` | 1× NVIDIA RTX 4090 24 GB, bf16, one request at a time | input cap 131,072 tokens, at most 676 options per choice; no truncation, 0 unsupported |
