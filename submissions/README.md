# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [RSI-Jev v6.1-VL 4B](https://huggingface.co/shgao/rsi-jev-v6.1-vl-4b) | 0.3 | 50.98 | [scores.json](https://huggingface.co/datasets/shgao/rsi-jev-benchmarks/blob/ed00bd76e06e676d9b40f29e9f4cdbb877715523/decision-index/runs/RSI-Jev-v6.1-VL-4B-0.3/scores.json) | `http` against `rsi-jev serve v6.1-vl-4b` from [Shanghua-Gao/RSI-Jev](https://github.com/Shanghua-Gao/RSI-Jev/tree/dd8b837eec6b05a417bb7e45cb97093d378d8c3f) @ `dd8b837`, effort unset (default); runner `87d4650` (0.2.1 requests) and `62d2f51` (0.3 GSM8K requests), scorer `62d2f51` | 1× NVIDIA H200 (16,000 of the 0.2.1 requests), 1× RTX PRO 6000 Blackwell (the other 134,759 0.2.1 requests and the 2,638 GSM8K requests); bf16 tower, fp32 scorer, one request at a time | 32,768 tokens per question (state + instructions + options); 5,120 options; no truncation (0 reached) |
