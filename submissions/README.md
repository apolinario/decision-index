# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [Saif 4B v2.1 (4B)](https://huggingface.co/GyrTechnology/saif-4b/tree/8319c9c33e88850b2e403378d2d0105957b38a37) by GyrTechnology (based on Qwen3.5-4B) | 0.3 | 62.94 | [scores.json](https://huggingface.co/datasets/GyrTechnology/decision-index-results/blob/7fc63b1f13688f44e9cd3171ad911b4e660125ce/runs/saif-4b-v2.1/scores.json) | `decision_index_engine:SaifEngine` (in the model repo, sha256 `9a0af82e`); kit `runner.py` sha256 `f4c9f36c` (see [saif-4b-v2.1](saif-4b-v2.1/README.md)) | 1 x NVIDIA RTX PRO 6000 Blackwell Max-Q (96 GB) | None; 0 unsupported, nothing truncated |
