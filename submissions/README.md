# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [ezjev-4b-s2](https://huggingface.co/everettjf/ezjev-4b-s2) | 0.2.1 | 51.15 | [scores.json](https://huggingface.co/datasets/everettjf/decision-index-results-ezjev-4b-s2/blob/603d2ffc15af8bb47c183ac586a141ef6f4c5263/runs/ezjev-4b-s2/scores.json) | `http` → llm2jev (chat prompt, T=1.26) over vLLM 0.30.0; runner/scorer `87d4650` | 1× NVIDIA RTX PRO 6000 (HF Jobs), bf16, one request at a time | `--max-model-len 131072`; no truncation, 0 unsupported |
