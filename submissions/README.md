# Decision Index submissions

Community submissions. Each line links a model to its complete run directory on the Hub (`runs/<name>/scores.json` must be present and say `"complete": true`; runs are re-scored on review). See the main README ("Submitting a model to the leaderboard") for the procedure. For the 0.3 Full score the maintainers additionally run the submitted model themselves on the private parts, exactly as specified below: weights must be public and the inference code runnable.

| Model | Results (scores.json) | Engine / commit | Hardware | Inference settings |
|---|---|---|---|---|
| [Tasksource-JEV-Nano-v1](https://huggingface.co/tasksource/tasksource-jev-nano-v1) (~149M, packdrop-s400) — Decision Index 0.3 public: 10.84 Balanced Skill, 33.12 Balanced Raw | [tasksource/decision-index-results/runs/official_03_packdrop_s400](https://huggingface.co/datasets/tasksource/decision-index-results/tree/main/runs/official_03_packdrop_s400) | `decision_models.eval.decision_index_official_engine:LateOnDecisionIndexEngine`, train_jev `7cb822a` (package `decision_models` since `bdd5227`) | 1× NVIDIA A10 23GB | isolated S+Qi; gamma=0.0; learned temperature 0.3168; max_context_length=2048; no truncation (`Unsupported` counts as wrong, 1 error / 153,397) |
