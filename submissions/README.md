# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [vera-core-2](https://huggingface.co/scar-ai/vera-core-2/tree/cfc7812b04b1c3e878e2e081b4ac39037d223d09) (Qwen3.5-2B, Apache-2.0) | 0.3 | 41.28 | [scores.json](https://huggingface.co/datasets/scar-ai/decision-index-results/blob/fe6e3c7bd626b174af5af7d62d16e8dd03812dda/runs/vera-core-2/scores.json) | `http` against `vera-serve --model scar-ai/vera-core-2` from `vera-s1[fast]==0.1.4` (default settings, see [vera-core-2/README.md](vera-core-2/README.md)); kit `9eb2dbe`, rescores identically with `f5a6e4f`; one request at a time | 1× AMD MI300X 192 GB (ROCm 7.1) | none: every request read whole, 0 unsupported, 0 errors |
