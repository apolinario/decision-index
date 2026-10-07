# Model submissions

| Model | Edition | Public index | Complete results | Engine / commit | Hardware | Inference settings | Declared limits |
|---|---|---:|---|---|---|---|---|
| [Saracura PT-BR 4B](https://huggingface.co/felhen-ai/saracura-ptbr-4b) | 0.3 | 37.96 | [scores.json](https://huggingface.co/datasets/felhen-ai/decision-index-results/blob/a7eebd8bc26e229b24973c3021989ff7e6f125fc/runs/saracura-ptbr-4b/scores.json) | `http` → `kev.serve` ([jaredpalmer/kev](https://github.com/jaredpalmer/kev)); runner/scorer `9eb2dbe` | 1× NVIDIA RTX 5090 32 GB, bf16, one request at a time | `kev.serve --run felhen-ai/saracura-ptbr-4b`, server defaults, `model=kev-latest` | `SERVE_MAX_STATE=65536` (Kev default); no truncation, 0 unsupported |
