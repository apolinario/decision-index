# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [Sieve-27B-v2](https://huggingface.co/sthanika-ai/Sieve-27B-v2/tree/f84ebfaca4073850add1f37eec4f3474d40093bb) | 0.3 | 68.02 | [scores.json](https://huggingface.co/datasets/sthanika-ai/Sieve-27B-v2-decision-index-results/blob/8b5499a5fa2fc68f47823c7d69d89416306d57c8/runs/Sieve-27B-v2/scores.json) | [`sieve.decision_index_engine:SieveEngine`](https://github.com/sthanika-ai/Sieve/blob/5f1e35791c64aaa39c76d4b20b79cc97490e95e2/sieve/decision_index_engine.py) @ `5f1e357`; kit `f5a6e4f` (0.3) | 2× NVIDIA B300 SXM6 288GB, suite in 6 shards, 3 engine processes per GPU; bf16 eager, fine-tuned layers copied into the backbone, one request at a time per process | 65,536 state tokens; 32,768 tokens per question with its options; up to 255 choice options per question; no truncation (none reached: 0 unsupported) |
