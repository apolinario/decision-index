# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [vera-large](https://huggingface.co/scar-ai/vera-large/tree/cc5adb16238d9ebae887da71f994cd04963d251f) (Qwen3.8-27B, Apache-2.0) | 0.3 | 67.75 | [scores.json](https://huggingface.co/datasets/scar-ai/decision-index-results/blob/296c909d7bc85834ddc746c5e2718e9b163299ec/runs/vera-large/scores.json) | `http` against `vera-serve --model scar-ai/vera-large` from `vera-s1[fast]==0.1.6` (see [vera-large/README.md](vera-large/README.md)); kit `9eb2dbe`, rescores identically with `25c6ec7`; one request at a time | 1× AMD MI300X 192 GB (ROCm 7.1) | none: every request read whole, 0 unsupported, 0 errors |
