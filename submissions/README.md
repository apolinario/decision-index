# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [RSI-Jev v7.0-VL 3B](https://huggingface.co/shgao/rsi-jev-v7.0-vl-3b) | 0.3 | 60.01 | [scores.json](https://huggingface.co/datasets/shgao/rsi-jev-benchmarks/blob/263d9ad3deddffeb00f85009c8a5a325802d3fd3/decision-index/runs/RSI-Jev-v7.0-VL-3B-0.3/scores.json) | `http` against `rsi-jev serve` of the v7.0-VL 3B package from [Shanghua-Gao/RSI-Jev](https://github.com/Shanghua-Gao/RSI-Jev/tree/479343b27946e2a44a8bb72e9fc4dc09a824cf20) @ `479343b`, weights byte-identical to HF tag `v7.0-VL-3B`, effort unset (default); runner `87d4650` (0.2.1 requests and the 0.3 GSM8K requests), 0.3 scoring (same code as the released 0.3 kit, `9eb2dbe`) | 1× RTX PRO 6000 Blackwell (all 153,397 requests); bf16 tower, fp32 scorer, one request at a time | 32,768 tokens per question (state + instructions + options); 5,120 options; no truncation (0 reached) |
