# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [Crivo-150M](https://huggingface.co/tardellirs/crivo-150m/tree/677d106fe58d74c324ad2ff18d0919c04113dd83) | 0.3 | 36.39 | [scores.json](https://huggingface.co/datasets/tardellirs/decision-index-results/blob/4d0016c1feb37f7609adbb6e90c0b45fc92a9736/runs/crivo-150m/scores.json) | `decision_index_engine:DecisionEngine` (in the model repo); kit `9eb2dbe` | 1× NVIDIA A100 40 GB, bf16 autocast, one request at a time | trained to 8,192 tokens, reads up to 32,768 (no truncation); 0 unsupported |
