# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [Sieve-27B](https://huggingface.co/sthanika-ai/Sieve-27B/tree/7836ff5604cb6682c835745fffd1b03a5ce4145c) | 0.3 | 60.04 | [scores.json](https://huggingface.co/datasets/sthanika-ai/Sieve-27B-decision-index-results/blob/85928c3be7fed4e10f8c30629da21456fa069267/runs/Sieve-27B/scores.json) | [`sieve.decision_index_engine:SieveEngine`](https://github.com/sthanika-ai/Sieve/blob/5284de70197642dbb4d55b4cf05a690b2955b080/sieve/decision_index_engine.py), `5284de70197642dbb4d55b4cf05a690b2955b080`; kit `62d2f51` (0.3) | 2× NVIDIA A100 80GB PCIe, two shards (one per GPU); single process per GPU, bf16 eager, one request at a time | 65,536 state tokens; 32,768 question tokens; up to 255 choice options; no truncation (none reached: 0 unsupported) |
