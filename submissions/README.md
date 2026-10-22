# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [tasksource-decider-nano](https://huggingface.co/tasksource/tasksource-decider-nano/tree/0dd9e308c2bcc40cd8404e78ed6dc64828fa326a) v5 (replaces v4) | 0.3 | 33.51 | [scores.json](https://huggingface.co/datasets/tasksource/decision-index-results/blob/49d7bcbbf84f2a57b1463f36a724d11c045313f7/runs/tasksource-decider-nano/scores.json) | `decision_index_engine:DeciderEngine` (in the model repo); kit `9eb2dbe` | 1× NVIDIA A30 24 GB (shared), bf16 autocast, one request at a time | 8,192-token context, no truncation, 7 unsupported (pairs above 8,192 tokens) |
