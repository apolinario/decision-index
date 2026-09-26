# Submissions

One row per submitted model. See the repository README, "Submitting a model to the leaderboard".

| Model | Decision Index | Results | Engine and exact code | Hardware | Declared capacity limits |
|---|---:|---|---|---|---|
| Jebadiah 27B (`frontier-infra/jebadiah-27b` @ `a5d7c80`, merged LoRA on `Qwen/Qwen3.8-27B` @ `1d4bf0f2`) | not run (run request) | compatibility pass only: `submissions/jebadiah-27b/compat/` | `jebadiah_engine:JebadiahEngine` (in `submissions/jebadiah-27b/`), wrapping the model repo's `scripts/` unchanged; kit `19ad28e` | our pass: Apple M3 Ultra (MPS, bf16); full run requested on 1 x RTX PRO 6000 | Context 262,144 tokens; up to 588 options per question; nothing truncated, no options removed |
