# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| Sieve-9B-v3 | 0.3 | 67.60 | [scores.json](https://huggingface.co/datasets/sthanika-ai/Sieve-9B-v3-eval-results/blob/089745f78f8733615304148bbbf0841b124fc090/runs/Sieve-9B-v3/scores.json) | [`sieve.decision_index_engine:SieveEngine`](https://github.com/sthanika-ai/Sieve/blob/223b92a5e5b5b1e1def38e1d53625a7244f6b569/sieve/decision_index_engine.py) @ `223b92a` (`sieve` 0.5.1; the run: our in-process letter engine, identical to its exact mode `SIEVE_LETTERS_PREFIX_MIN=0`); kit `25c6ec7` (0.3) | 10× NVIDIA A100-SXM4-40GB, suite in 10 shards, one engine process per GPU; bf16 eager, one request at a time per process | no token limit; up to 255 choice options per question; no truncation (0 unsupported) |
