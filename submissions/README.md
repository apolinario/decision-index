# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [RSI-Jev v5.0-VL 3B](https://huggingface.co/shgao/rsi-jev-v5.0-vl-3b) | 0.2.1 | 38.38 | [scores.json](https://huggingface.co/datasets/shgao/rsi-jev-benchmarks/blob/9291bf112b3325cc341f47f497835b057faee5bc/decision-index/runs/RSI-Jev-v5.0-VL-3B/scores.json) | `http` against `rsi-jev serve` from [Shanghua-Gao/RSI-Jev](https://github.com/Shanghua-Gao/RSI-Jev/tree/c6b72b09f263a5f90cec0db94c109a3f6fc6b303) @ `c6b72b0` (pre-release build, see notes); runner/scorer `87d4650` | 1× NVIDIA RTX PRO 6000 Blackwell, bf16 tower, one request at a time | 32,768 tokens per question (state + instructions + options); 5,120 options; no truncation (0 reached) |
