# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [jiwo-0.8b](https://huggingface.co/eljiwo/jiwo-0.8b) | 0.2.1 | 28.00 | [scores.json](https://huggingface.co/datasets/eljiwo/decision-index-results/blob/d2caa6b3d8ebe0f764bf7d1603c6d512a7d2d7ca/runs/jiwo-0.8b/scores.json) | `http` → `jiwo serve` ([jiwidi/jiwo](https://github.com/jiwidi/jiwo)); runner/scorer `87d4650` | 1× NVIDIA H100 80 GB, bf16, one request at a time | `JIWO_MAX_LENGTH=65536`; no truncation, 0 unsupported |
| [jiwo-4b](https://huggingface.co/eljiwo/jiwo-4b) | 0.2.1 | 44.97 | [scores.json](https://huggingface.co/datasets/eljiwo/decision-index-results/blob/d2caa6b3d8ebe0f764bf7d1603c6d512a7d2d7ca/runs/jiwo-4b/scores.json) | `http` → `jiwo serve` ([jiwidi/jiwo](https://github.com/jiwidi/jiwo)); runner/scorer `87d4650` | 1× NVIDIA H100 80 GB, bf16, one request at a time | `JIWO_MAX_LENGTH=65536`; no truncation, 0 unsupported |
