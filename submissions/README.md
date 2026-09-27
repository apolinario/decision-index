# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [Jet](https://huggingface.co/michaljach/jet/tree/fbc3d2daa679e0d4bd9f99c9912b6496d5a41f0a) (v6.2) | 0.2.1 | 42.61 | [scores.json](https://huggingface.co/datasets/michaljach/jet-decision-index-results/blob/ef10608bc9ec81c64dcabb881ca8c2f46f0e7ffd/runs/jet-v6.2/scores.json) | [Native Jet engine](https://github.com/michaljach/decision-index/blob/21092e85a7a2b641b23830ffd050af89168c4a6e/submissions/jet/engine.py), `21092e8`; runner/scorer `87d4650` | 1× NVIDIA RTX 4080 SUPER 16GB | 16,384 complete prompt tokens; 2–255 choice options; no truncation |
