# Submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| Faruq-32M ([`FRQONE/Faruq-32M`](https://huggingface.co/FRQONE/Faruq-32M) @ `0c416ea`) | 0.3 | 26.87 | [`runs/faruq-32m`](https://huggingface.co/datasets/FRQONE/faruq-decision-index-results/tree/d9eeaae73a011c8de517335e85121f102df0dd5a/runs/faruq-32m) | in-process `faruq_engine:FaruqKitEngine` from the model repo (weights `stage-all`, bf16 autocast, deterministic), kit `25c6ec7` (text code) | RTX 5090 32GB, one request at a time | 7,999 tokens per question; a request holding a longer question is refused, never truncated |
