# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [Sieve-9B](https://huggingface.co/sthanika-ai/Sieve-9B/tree/dc2208c47c34afbfc78c393d6e738634b66b4e27) | 0.2.1 | 41.71 | [scores.json](https://huggingface.co/datasets/sthanika-ai/Sieve-9B-decision-index-results/blob/99c8230ed11da919860b074f447e67f43bb9dce9/runs/Sieve-9B/scores.json) | [`sieve.decision_index_engine:SieveEngine`](https://github.com/sthanika-ai/sieve/blob/28aed61bb1f29919f883823acf708b45e1cdb10f/sieve/decision_index_engine.py), `28aed61bb1f29919f883823acf708b45e1cdb10f`; runner/scorer `87d4650` | 1× NVIDIA A100 80GB PCIe (shared), single process, bf16 eager | 65,536 state tokens; 32,768 question tokens; up to 255 choice options; no truncation (none reached: 0 unsupported) |
