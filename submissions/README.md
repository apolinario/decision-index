# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [crystal-0.8b](https://huggingface.co/Kyunghwan-kim/crystal-0.8b/tree/befc98ecaf858a5e4cafc11b8921fea709092943) | 0.3 | 42.79 | [scores.json](https://huggingface.co/datasets/Kyunghwan-kim/crystal-0.8b-decision-index/blob/bb6fdaa908d5adf6b7d0e780c82bdcd7cfcc01e6/runs/crystal-0.8b/scores.json) | [`crystal.engine:CrystalEngine`](https://huggingface.co/Kyunghwan-kim/crystal-0.8b/blob/befc98ecaf858a5e4cafc11b8921fea709092943/crystal/engine.py) @ `befc98e`, BF16 backbone, FP32 readout, noncausal full-attention layers, `batch_tokens=32768`, `max_batch=8`; runner and scorer `f5a6e4f` | 1× RTX A6000, one request at a time | text only, ≤255 options per question; none reached, nothing truncated, none unsupported |
