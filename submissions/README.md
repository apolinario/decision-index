# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [OneJev1.6g-31B](https://huggingface.co/OmniJev/OneJev1.6g-31B/tree/9334b7c0923c813483e2ed6eb688904c50b3901a) | 0.3 | 69.82 | [scores.json](https://huggingface.co/datasets/OmniJev/onejev-decision-index-results/blob/cc6ffce9944ef9ede7ce97af8b1a6a8c22c9f375/runs/onejev1.6g-31b/scores.json) | kit `http` engine against `flide serve` ([Flide](https://github.com/OmniJev/Flide) 0.1.7, PyPI), bf16, `--planners 8 --max-branch-tokens 131072 --max-request-tokens 262144`; runner and scorer `25c6ec7` | 2x H200, eight requests at a time per server | 131,072 tokens per question branch; none reached, nothing truncated, none unsupported |
