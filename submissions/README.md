# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [Sieve-4B](https://huggingface.co/sthanika-ai/Sieve-4B/tree/fdad222de6c018a8c3613abc623c9005d76657c7) | 0.3 | 50.06 | [scores.json](https://huggingface.co/datasets/sthanika-ai/Sieve-4B-decision-index-results/blob/ce56e86cf4ed9a20edb3d1cb904dafe1698c6d79/runs/Sieve-4B/scores.json) | `http` engine against `python -m sieve4b.server` (code in the model repo, `sieve4b/`, same revision); kit `9eb2dbe` (0.3) | 2× NVIDIA A100 80GB PCIe, 6 server processes (3 per GPU), suite in 6 shards; bf16, adapter merged, one request at a time per process | up to 255 choice options, 10 score levels, 32,768 tokens for state + longest question, 65,536 per request; no truncation (none reached: 0 unsupported) |
