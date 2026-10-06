# Decision Index submissions

Community submissions. Each line links a model to its complete run directory on the Hub (`runs/<name>/scores.json` must be present and say `"complete": true`; runs are re-scored on review). See the main README ("Submitting a model to the leaderboard") for the procedure.

| Model | Results (scores.json) | Engine / commit | Hardware |
|---|---|---|---|
| [Tasksource-JEV-Nano-v1](https://huggingface.co/tasksource/tasksource-jev-nano-v1) (~149M, isolated S+Qi, packdrop-s400) — Decision Index 0.2.1: 10.46 Balanced Skill, 32.09 Balanced Raw | [tasksource/decision-index-results/runs/official_full_packdrop_s400_isolated](https://huggingface.co/datasets/tasksource/decision-index-results/tree/main/runs/official_full_packdrop_s400_isolated) | `modernjev.eval.decision_index_official_engine:LateOnDecisionIndexEngine` (isolated), train_jev `7cb822a` (engine `b6d10b1`) | 1× NVIDIA A30 24GB |
