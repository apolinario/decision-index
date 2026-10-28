# Submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| Faruq-150M ([`FRQONE/Faruq-150M`](https://huggingface.co/FRQONE/Faruq-150M) @ `c53e9ac`) | 0.3 | 35.62 | [`runs/faruq-150m`](https://huggingface.co/datasets/FRQONE/faruq-decision-index-results/tree/21a987c41075f8cbbbf377cbcf59cdd075c46667/runs/faruq-150m) | in-process `faruq_engine:FaruqKitEngine` from the model repo (weights `stage-all`, bf16 autocast, deterministic), kit `25c6ec7` (text code) | RTX 5090 32GB, one request at a time | 7,999 tokens per question; a request holding a longer question is refused, never truncated |
