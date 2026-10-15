# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [Sieve-2B-v3](https://huggingface.co/sthanika-ai/Sieve-2B-v3/tree/905613c41f81fe4cc249d42c5f6b06d1e1c2337f) | 0.3 | 53.51 | [scores.json](https://huggingface.co/datasets/sthanika-ai/Sieve-2B-v3-decision-index-results/blob/e53d0854b6c944eb35d88b7d673841732d5cfc62/runs/Sieve-2B-v3/scores.json) | [`sieve.decision_index_engine:SieveEngine`](https://github.com/sthanika-ai/Sieve/blob/943f833136e69a4321cfdd6bfd7b4f338fc17840/sieve/decision_index_engine.py) @ `943f833`; kit `f5a6e4f` (0.3) | 2× NVIDIA A100 80GB PCIe, suite in 6 shards, 3 engine processes per GPU; bf16 eager, fine-tuned layers merged into the backbone, one request at a time per process | 65,536 state tokens; 32,768 tokens per question with its options; up to 255 choice options per question; no truncation (none reached: 0 unsupported) |
