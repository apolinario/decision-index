# Submissions

One row per submitted model. See the repository README, "Submitting a model to the leaderboard".

| Model | Decision Index | Results | Engine and exact code | Hardware | Declared capacity limits |
|---|---:|---|---|---|---|
| Jebadiah 9B v1 (`frontier-infra/jebadiah-9b-v1` @ `ea86ab5`, LoRA on `Qwen/Qwen3.5-9B-Base` @ `68c46c4b`) | not run (run request) | compatibility pass only: `submissions/jebadiah-9b-v1/compat/` | `jebadiah_engine:JebadiahEngine` (in `submissions/jebadiah-9b-v1/`), wrapping the model repo's `scripts/` unchanged; kit `19ad28e` | our pass: Apple M3 Ultra (MPS, bf16); full run requested on 1 x RTX PRO 6000 | Context 262,144 tokens; up to 588 options per question; nothing truncated, no options removed |
