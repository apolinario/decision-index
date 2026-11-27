# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [Sieve-31B-v4](https://huggingface.co/sthanika-ai/Sieve-31B-v4/tree/464dc7f46e150407aec322634d321071ba4b433f) | 0.3 | 72.01 | [scores.json](https://huggingface.co/datasets/sthanika-ai/Sieve-31B-v4-decision-index-results/blob/8ca746143135e0e1eb390ae58d7c7439a5b3b786/runs/Sieve-31B-v4/scores.json) | [`sieve.decision_index_engine:SieveEngine`](https://github.com/sthanika-ai/Sieve/blob/aafc31b781e0f590d180d2b1c8ade283d1d8401a/sieve/decision_index_engine.py) @ `aafc31b` (`sieve` 0.5.2); kit `f5a6e4f` (0.3) | NVIDIA A100-SXM4-40GB, suite in 56 shards, each engine process splitting the model over 2 or 4 GPUs (`devices` option); bf16 eager, one request at a time per process | no token limit; up to 255 choice options per question; no truncation (0 unsupported) |
