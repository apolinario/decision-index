# Submissions

One row per submitted model. See the repository README, "Submitting a model to the leaderboard".

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [JEV-9B](https://huggingface.co/autotrust/JEV-9B/tree/4ab5dfb9331c4eb3a212742e1a1aa5446c1fda35) (`autotrust/JEV-9B` @ `4ab5dfb`; LoRA + 24-slot decision head on `Qwen/Qwen3.5-9B`) | 0.2.1 | 43.14 | [scores.json](https://huggingface.co/datasets/autotrust/jev-decision-index-results/blob/427cd51cf1e163eaebcb7fd98c37e0412bc882f0/runs/jev-9b/scores.json) | `jev_engine:JevEngine` (`submissions/jev/`); kit runner `87d4650` for rows 1–35,590, batched driver `submissions/jev/fast_run.py` (same engine, 64 requests per batch) for the rest | 1 × NVIDIA B200 | Context 262,144 tokens; 16 choice options per pass, wider questions (up to 255) read in groups of ≤ 16 plus a final, no option removed; no truncation |
