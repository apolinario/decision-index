# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware / how run | Declared limits |
|---|---|---:|---|---|---|---|
| [Duumvir Decider](https://huggingface.co/tardellirs/duumvir-decider/tree/588eedb63a4a8b3451fcf851d5e95a29c357e40f) (new) | 0.3 | 75.42 | [scores.json](https://huggingface.co/datasets/tardellirs/duumvir-decider-decision-index/blob/75e52db383e49b92cf24d1a1e69b37037d65e7a6/runs/duumvir-decider/scores.json) | `duo_engine:KarpoKnowLineCascade` (in the model repo); kit `25c6ec7` | complete run of the package (kit pipeline), 1× NVIDIA A100-SXM4 80 GB, bf16, one request at a time | text only; karpo reads up to 32,768 tokens, longer requests go to the KnowLine side (no truncation); 0 unsupported |
