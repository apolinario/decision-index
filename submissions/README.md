# Submissions

| Model | Decision Index | Results | Engine / code | Hardware | Declared capacity limits |
|---|---|---|---|---|---|
| zev-latest | 4.76 | [bkhubbard/decision-index-results](https://huggingface.co/datasets/bkhubbard/decision-index-results) · `runs/zev-latest/scores.json` (complete: true) | `http` engine, kit `87d4650` · pure Rust zero-token decision engine [zev-rs](https://github.com/bhubbard/zev-rs) @ `f1d17c2` (v0.3.2), `/v1/systemone` · 0.5 ms median latency | Apple M3 Pro (12-core CPU) | `max_state_bytes`: 2MB, `max_questions`: 64, `max_slots`: dynamic (all candidate criteria scored, nothing truncated) |
| zev-gemma | 4.76 | [bkhubbard/decision-index-results](https://huggingface.co/datasets/bkhubbard/decision-index-results) · `runs/zev-gemma/scores.json` (complete: true) | `http` engine, kit `87d4650` · speculative hybrid decision engine [zev-rs](https://github.com/bhubbard/zev-rs) @ `f1d17c2` + Gemma distillation fallback, `/v1/systemone` · 0.6 ms median latency | Apple M3 Pro (12-core CPU) | `max_state_bytes`: 2MB, `max_questions`: 64, `max_slots`: dynamic (all candidate criteria scored, nothing truncated) |
