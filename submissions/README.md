# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [RSI-Jev v4.0-VL 2B](https://huggingface.co/shgao/rsi-jev-v4.0-vl-qwen3.5-2b) | 0.2.1 | 28.31 | [scores.json](https://huggingface.co/datasets/shgao/rsi-jev-benchmarks/blob/a67ec735ab000e977d7bd327206a4e8c2e4e3ac1/decision-index/runs/RSI-Jev-v4.0-VL/scores.json) | `http` against `rsi-jev serve` from [Shanghua-Gao/RSI-Jev](https://github.com/Shanghua-Gao/RSI-Jev/tree/83ed4a3b60d21ddea3bc94247e1ca64caa64ad2f) @ `83ed4a3`; runner/scorer `87d4650` | 1× NVIDIA RTX PRO 6000 Blackwell, bf16 tower, one request at a time | 32,768 tokens per question (state + instructions + options); 5,120 options; no truncation (0 reached) |
