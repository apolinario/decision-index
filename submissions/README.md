# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [Sieve-2B](https://huggingface.co/sthanika-ai/sieve-2b/tree/5aa4d83a4096bcb6d0e0ae75738a225a8fad4b1e) | 0.2.1 | 21.74 | [scores.json](https://huggingface.co/datasets/sthanika-ai/Sieve-2B-decision-index-results/blob/254be32b79839903ec000152ac0226c12585be6f/runs/Sieve-2B/scores.json) | [`sieve_engine:SieveEngine`](https://huggingface.co/sthanika-ai/sieve-2b/blob/5aa4d83a4096bcb6d0e0ae75738a225a8fad4b1e/sieve_engine.py), `5aa4d83`; runner/scorer `87d4650` | 2× NVIDIA A100 80GB PCIe (shared), 7 processes over 24 shards, bf16 | 32,768 tokens for state + question + longest option; no option-count limit; no truncation (0 reached); duplicate option texts refused (2 scored requests) |
