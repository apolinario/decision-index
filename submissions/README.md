# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [tasksource-decider-nano](https://huggingface.co/tasksource/tasksource-decider-nano/tree/6fa4fce083f71aaa46feec30aa2a60a1ae359ee4) (replaces Tasksource-JEV-Nano-v1) | 0.3 | 29.05 | [scores.json](https://huggingface.co/datasets/tasksource/decision-index-results/blob/5853dd01c0cebaf96794b6f8c491f277e743e9c1/runs/tasksource-decider-nano/scores.json) | `decision_index_engine:DeciderEngine` (in the model repo); kit `9eb2dbe` | 1× NVIDIA A10 23 GB, bf16 autocast, one request at a time | 8,192-token context, no truncation, 7 unsupported (pairs above 8,192 tokens) |
