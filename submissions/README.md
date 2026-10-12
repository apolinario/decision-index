# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [Sieve-2B-v2](https://huggingface.co/sthanika-ai/Sieve-2B-v2/tree/b28bd2a6d284a0ce712bed7f6f098e714d3181d2) | 0.3 | 37.26 | [scores.json](https://huggingface.co/datasets/sthanika-ai/Sieve-2B-v2-decision-index-results/blob/f0cda57632a4e002d71f8897af9e2d346c126373/runs/Sieve-2B-Plus/scores.json) | [`sieve_engine:SieveEngine`](https://huggingface.co/sthanika-ai/Sieve-2B-v2/blob/b28bd2a6d284a0ce712bed7f6f098e714d3181d2/sieve_engine.py) @ `b28bd2a`, `max_batch_tokens=49152`, `mem_fraction=0.16`, bf16; runner `87d4650` (0.2.1 requests) and `62d2f51` (0.3 GSM8K requests), scorer `62d2f51` | 2× A100 80GB PCIe, 10 processes over 24 shards (0.2.1 requests); 1× A100 80GB PCIe (0.3 GSM8K requests) | 32,768 tokens for state + question + longest option; no option-count limit; no truncation (0 reached); duplicate option texts refused (2) |
