# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [Aplomb 1](https://huggingface.co/empiriolabsai/aplomb-1/tree/d67432916c6e1fe569ab2de7942755eee91e6880) | 0.3 | 43.49 | [scores.json](https://huggingface.co/datasets/empiriolabsai/decision-index-results-aplomb-1/blob/490b0a103e0c4febacdb56b6456b12f34e5426d2/runs/aplomb-1-0.3/scores.json) | kit `http` engine against `serve_aplomb.py` from the model repo (reference inference, bf16, `--debias`, one request at a time); runner and scorer `9eb2dbe` (same tree as `62d2f51`) | 16 × NVIDIA RTX PRO 6000 Blackwell, one per machine; 16 shards by a hash of `run_id`, each request once | 1M tokens of state plus question, 255 options per choice, 512 questions per request; none reached, 0 unsupported, no truncation |
