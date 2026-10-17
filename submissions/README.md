# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [Prumo](https://huggingface.co/tardellirs/prumo/tree/b445c46915ef5439fc4239e06f4cceee963954a4) | 0.3 | 20.14 | [scores.json](https://huggingface.co/datasets/tardellirs/decision-index-results/blob/a54aa7e6569d705db24aa1bf439f49754db8dcfe/runs/prumo/scores.json) | `decision_index_engine:DecisionEngine` (in the repo); kit `9eb2dbe` | 1 CPU thread (Apple M2), no GPU, one request at a time | rule-based, 0 learned parameters (~700 hand-written constants); 0 unsupported, no truncation |
