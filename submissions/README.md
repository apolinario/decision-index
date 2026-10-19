# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [jdi-ptr14-0.8b](https://huggingface.co/AnamK/jdi-ptr14-0.8b/tree/5bb55cb177d4f0d344f891af146775be30dbb3c5) (Qwen3.5-0.8B, CC BY-NC 4.0) | 0.3 | 35.79 | [scores.json](https://huggingface.co/datasets/AnamK/decision-index-results/blob/20844ea936c62259276fba4502abac7bcdf3ebd3/runs/ptr14nosyn-soup3/scores.json) | in-process `jdi_ptr14.engine:Ptr14Engine` (in the model repo under `engine/`; settings in [jdi-ptr14-0.8b/README.md](jdi-ptr14-0.8b/README.md)); kit `9eb2dbe`; one request at a time | 1× NVIDIA RTX 3090 24 GB (shared) | context limit 262,144 tokens (longer = unsupported, none in the suite); 0 unsupported, 0 errors, no truncation |
