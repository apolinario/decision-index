# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [tasksource-decider-nano](https://huggingface.co/tasksource/tasksource-decider-nano/tree/afa49af5a6b3550460fafc431216fbf1e8978fdb) (replaces Tasksource-JEV-Nano-v1) | 0.3 | 26.77 | [scores.json](https://huggingface.co/datasets/tasksource/decision-index-results/blob/98b8ddb74c91f2d80e5c1cb6bbccd25aea7130d3/runs/tasksource-decider-nano/scores.json) | `decision_index_engine:DeciderEngine` (in the model repo); kit `9eb2dbe` | 1× NVIDIA A10 23 GB, bf16 autocast, one request at a time | 8,192-token context, no truncation, 7 unsupported (pairs above 8,192 tokens) |
