# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [Sieve-9B-Plus](https://huggingface.co/sthanika-ai/Sieve-9B-Plus/tree/4a5a79ea5e92d0830da936307823d11acc78ef1b) | 0.3 | 53.35 | [scores.json](https://huggingface.co/datasets/sthanika-ai/Sieve-9B-Plus-decision-index-results/blob/ec7fa8cba67edbfa32345b471e755e7f044f7026/runs/Sieve-9B-Plus/scores.json) | [`sieve.decision_index_engine:SieveEngine`](https://github.com/sthanika-ai/Sieve/blob/e632700ddc2f25336bf9beace1ef32765db3e6df/sieve/decision_index_engine.py), `e632700ddc2f25336bf9beace1ef32765db3e6df`; kit `62d2f51` (0.3) | 2× NVIDIA H100 80GB HBM3, suite in 4 shards, 2 engine processes per GPU; bf16 eager, adapter merged, one request at a time per process | 65,536 state tokens; 32,768 question tokens; up to 255 choice options; no truncation (none reached: 0 unsupported) |
