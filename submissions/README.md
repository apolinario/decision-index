# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [tasksource-decider-nano](https://huggingface.co/tasksource/tasksource-decider-nano/tree/46e88e08cc1a3122a348ce134f18865e93f8e1c8) (replaces Tasksource-JEV-Nano-v1) | 0.3 | 30.90 | [scores.json](https://huggingface.co/datasets/tasksource/decision-index-results/blob/78ee238948680790d99fb8ba66fbde29a4ddaacd/runs/tasksource-decider-nano/scores.json) | `decision_index_engine:DeciderEngine` (in the model repo); kit `9eb2dbe` | 1× NVIDIA A10 23 GB, bf16 autocast, one request at a time | 8,192-token context, no truncation, 7 unsupported (pairs above 8,192 tokens) |
