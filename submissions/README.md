# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [RSI-Jev v7.0-VL 4B](https://huggingface.co/shgao/rsi-jev-v7.0-vl-4b) | 0.3 | 60.61 | [scores.json](https://huggingface.co/datasets/shgao/rsi-jev-benchmarks/blob/d0b48e473a83ec7d2eff48f3b34b372bde5d0ca5/decision-index/runs/RSI-Jev-v7.0-VL-4B-0.3/scores.json) | `http` against `rsi-jev serve v7.0-vl-4b` from [Shanghua-Gao/RSI-Jev](https://github.com/Shanghua-Gao/RSI-Jev/tree/479343b27946e2a44a8bb72e9fc4dc09a824cf20) @ `479343b`, released files (HF tag `v7.0-VL`), effort unset (default); runner `87d4650` (0.2.1 requests and the 0.3 GSM8K requests), 0.3 scoring (same code as the released 0.3 kit, `9eb2dbe`) | 1× RTX PRO 6000 Blackwell (all 153,397 requests); bf16 tower, fp32 scorer, one request at a time | 32,768 tokens per question (state + instructions + options); 5,120 options; no truncation (0 reached) |
