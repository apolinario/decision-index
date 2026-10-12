# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [Sieve-31B](https://huggingface.co/sthanika-ai/Sieve-31B/tree/29de6dd7e489ee01f05e35590494adf9400685a2) | 0.3 | 62.52 | [scores.json](https://huggingface.co/datasets/sthanika-ai/Sieve-31B-decision-index-results/blob/ee0b87ae181517661e7cdf8fde4d0f73c997ee5b/runs/Sieve-31B/scores.json) | [`sieve31b.engine:Sieve31BEngine`](https://huggingface.co/sthanika-ai/Sieve-31B/blob/29de6dd7e489ee01f05e35590494adf9400685a2/sieve31b/engine.py) in the model repo @ `29de6dd`; kit `62d2f51` (0.3) | Run: 2× NVIDIA A100 80GB PCIe (batched evaluator, same prompts/readout as the engine; probabilities at T=1 re-read at T 8.5495). Engine latency/parity checked on 1× RTX PRO 6000 Server, one request at a time | 255 options per question; 32,768 tokens per row (state + one question); no truncation (none reached: 0 unsupported) |
