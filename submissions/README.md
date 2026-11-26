# Submissions

One entry per model. Each run must be the complete public 0.3 suite (`scores.json` says
`"complete": true`) and untouched: the results file is re-scored on review. See the
"Submitting a model to the leaderboard" section of the top-level `README.md` for the full
requirements.

## Text board

### Zoellent xjyh-jev-1.3

```text
Model: Zoellent xjyh-jev-1.3
Weights: https://huggingface.co/Zoellent/xjyh-jev-1.3
Results: https://huggingface.co/datasets/Zoellent/decision-index-results/tree/main/runs/xjyh-jev-1.3
Kit: decision_index 0.3
Engine: decision_index.http, model=xjyh-jev-1.3, endpoint=/v1/systemone (StartLux-Prism v1.0.1, Q8_0 GGUF sha256 57779897aee8721f3349f4bcc9202065748f989cf619015b0bfee2682bb0b86e)
Hardware: 8 x NVIDIA H100 80GB (one Prism replica per GPU)
Settings: max_tokens=49152 (7 standard replicas) / max_tokens=81920 (1 big-context replica), mem_fraction=0.9 / 0.97, graph_max_tokens=8192, budget=8192, prefix_cache=0, think_mode=never, max_images=0; decision temperatures choice=1.027029, noul=0.996792, score=1.013066 (global 1.018182)
Declared limits: max context 49152 tokens (standard replica), 81920 tokens (big-context replica); max options 26; no image inputs; no truncation of state or options
Training data: train split only; Decision Index 0.3 suite rows excluded from training; no benchmark test split used

Public index: decision_index=57.36, raw_index=67.62, breadth_skill=56.72 (edition 0.3, suite release-v3, complete=true, coverage=1.0)
```

The results dataset is private by default (as produced by the kit's `hf-job`); access for
review can be granted on request.

Note: weights and results live under the `Zoellent` organization
(`Zoellent/xjyh-jev-1.3`, `Zoellent/decision-index-results`); the previous `haibin16`
URLs redirect to the new locations.
