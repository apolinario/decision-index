# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [jiwo-0.8b v2](https://huggingface.co/eljiwo/jiwo-0.8b/tree/71b7911cf68f7df52fdc2dbf07de1e158bb807f2) (updates jiwo 0.8B) | 0.3 | 29.58 | [scores.json](https://huggingface.co/datasets/eljiwo/decision-index-results/blob/960a30940884014b3b299167e73e257694416d61/runs/jiwo-0.8b-v2-0.3/scores.json) | `http` → `jiwo serve`; kit `62d2f51` (0.2.1 run with `87d4650`, resumed under 0.3); public code [jiwidi/jiwo@752d8c3](https://github.com/jiwidi/jiwo/tree/752d8c3) | 1× NVIDIA H100 80 GB, bf16, one request at a time | `JIWO_MAX_LENGTH=65536`; no truncation, 0 unsupported; fastest path: `JIWO_CUDA_GRAPHS=1` |
