# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [Prumo](https://huggingface.co/tardellirs/prumo/tree/f42441d7a1c2dc970c531d7c6aa89fa1ae264f38) · **size 527** (hand-set parameters in `model.safetensors`, 0 learned) | 0.3 | 20.14 | [scores.json](https://huggingface.co/datasets/tardellirs/decision-index-results/blob/a0bf30d99d257afe28db46f75e5fe0c60415ab5d/runs/prumo/scores.json) | `decision_index_engine:DecisionEngine` (in the repo); kit `9eb2dbe` | 1 CPU thread (Apple M2), no GPU, one request at a time | rule-based, 527 hand-set parameters, 0 learned; 0 unsupported, no truncation |
