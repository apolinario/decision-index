# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [CoCo-Decision-4B-Ko](https://huggingface.co/corners-ai/CoCo-Decision-4B-Ko) (`v1.0.0`) | 0.2.1 | 41.03 | [scores.json](https://huggingface.co/datasets/corners-ai/CoCo-Decision-4B-Ko-decision-index/blob/485635190dd8ab36a0cba4b7f4e8203cca07cf6a/runs/CoCo-Decision-4B-Ko/scores.json) | `http` → `omj serve` ([iamupd/oh-my-jev@f0b8c78](https://github.com/iamupd/oh-my-jev/commit/f0b8c78684b05b18137ec8d1b797dfdcd7463a7a), semif backend, bf16); runner/scorer `87d4650` | 1× NVIDIA GeForce RTX 5090 (32 GB) | prompts up to 32,768 tokens, up to 237 options per question; 0 unsupported, no truncation |
