# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [Crivo-400M](https://huggingface.co/tardellirs/crivo-400m/tree/81f9cab8d7b3cc049b33033952850bb15ce654ae) | 0.3 | 39.79 | [scores.json](https://huggingface.co/datasets/tardellirs/decision-index-results/blob/6d9fb5a6a10c71069442d50954a553dad280a9c8/runs/crivo-400m/scores.json) | `decision_index_engine:DecisionEngine` (in the model repo); kit `9eb2dbe` | 1× NVIDIA A100 40 GB, bf16 autocast, one request at a time | trained to 8,192 tokens, reads up to 32,768 (no truncation); 0 unsupported |
