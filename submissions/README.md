# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [OneJev1.5g-31B](https://huggingface.co/OmniJev/OneJev1.5g-31B/tree/20e5094ed6850616e12d0f1b311d178b453ae1f2) | 0.3 | 65.83 | [scores.json](https://huggingface.co/datasets/OmniJev/onejev-decision-index-results/blob/2e04dcca3afbbea622fef03e2b4277c6cc3fddbb/runs/onejev1.5g-31b/scores.json) | kit `http` engine against `flide serve` ([Flide](https://github.com/OmniJev/Flide) 0.1.7, PyPI), bf16, `--max-branch-tokens 131072 --max-request-tokens 262144`; runner and scorer `25c6ec7` | 2x H200, eight requests at a time per server | 131,072 tokens per question branch; none reached, nothing truncated, none unsupported |
