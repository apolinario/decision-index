# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [vera-core](https://huggingface.co/scar-ai/vera-core/tree/22bf59c482849e0ee51a36252ca03b778e94aa40) (Qwen3.5-2B) | 0.3 | 30.11 | [scores.json](https://huggingface.co/datasets/scar-ai/decision-index-results/blob/1929fc9f1ba7f60a31bc48d340debcfd97dfffa3/runs/vera-core/scores.json) | `http` against `vera-serve --no-caps` (vera-s1 0.1.3); kit `9eb2dbe` | 1× AMD MI300X 192 GB (ROCm 7.1), weights from the main revision (bf16), one request at a time | none: no truncation, 0 unsupported |
| [vera-spark](https://huggingface.co/scar-ai/vera-spark/tree/964a12f7a7c2aaa08950af899277de789f84045c) (ModernBERT-large, 400M) | 0.3 | 19.59 | [scores.json](https://huggingface.co/datasets/scar-ai/decision-index-results/blob/1929fc9f1ba7f60a31bc48d340debcfd97dfffa3/runs/vera-spark/scores.json) | `http` against `vera-serve --no-caps` (vera-s1 0.1.3); kit `9eb2dbe` | 1× AMD MI300X 192 GB (ROCm 7.1), weights from the main revision (bf16), one request at a time | none: no truncation, 0 unsupported |
