# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [OneJev1.7g-31B](https://huggingface.co/OmniJev/OneJev1.7g-31B/tree/ffc990bf23381c8b0a1773773bcafb33538f14ac) | 0.3 | 71.82 | [scores.json](https://huggingface.co/datasets/OmniJev/onejev-decision-index-results/blob/c04fbad9d2d4046a64a60328014abbd43054edcc/runs/onejev1.7g-31b/scores.json) | kit `http` engine against `flide serve` ([Flide](https://github.com/OmniJev/Flide) 0.1.7, PyPI), bf16, `--format template --planners 8 --max-branch-tokens 131072 --max-request-tokens 262144`; runner and scorer `25c6ec7` | 2x H200, twelve requests at a time per server | 131,072 tokens per question branch; none reached, nothing truncated, none unsupported |
