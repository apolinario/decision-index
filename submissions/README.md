# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [Crivo-32M](https://huggingface.co/tardellirs/crivo-32m/tree/bbf3f1924523f530b3a9e621f2f20f314ea462aa) | 0.3 | 25.06 | [scores.json](https://huggingface.co/datasets/tardellirs/decision-index-results/blob/eaf516f3d4eea9935cf8f3ae5d6688499498edd4/runs/crivo-32m/scores.json) | `decision_index_engine:DecisionEngine` (in the model repo); kit `9eb2dbe` | 1× NVIDIA A100 40 GB, bf16 autocast, one request at a time | trained to 8,192 tokens, reads up to 32,768 (no truncation); 0 unsupported |
