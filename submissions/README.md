# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [jiwo-4b v2](https://huggingface.co/eljiwo/jiwo-4b/tree/9adc559ff7ec5baffade3998f7bca85da0d124ed) (updates jiwo 4B) | 0.3 | 46.06 | [scores.json](https://huggingface.co/datasets/eljiwo/decision-index-results/blob/cdb86582a7117debc17da64a6e75032508c7e7ef/runs/jiwo-4b-v2-0.3/scores.json) | `http` → `jiwo serve` ([jiwidi/jiwo@40d43c4](https://github.com/jiwidi/jiwo/tree/40d43c4)); kit `62d2f51` (0.2.1 run with `87d4650`, resumed under 0.3) | 1× NVIDIA H100 80 GB, bf16, one request at a time | `JIWO_MAX_LENGTH=65536`; no truncation, 0 unsupported; fastest path: `JIWO_CUDA_GRAPHS=1` with [jiwo@752d8c3](https://github.com/jiwidi/jiwo/tree/752d8c3) |
