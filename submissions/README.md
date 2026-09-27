# Submissions

| Model | Decision Index | Results | Engine / code | Hardware | Declared capacity limits |
|---|---|---|---|---|---|
| zev-latest | 1.51 | [bkhubbard/decision-index-results](https://huggingface.co/datasets/bkhubbard/decision-index-results) · `runs/zev-latest/scores.json` (complete: true) | `http` engine, kit `87d4650` · pure Rust zero-token decision engine [zev-rs](https://github.com/bhubbard/zev-rs) @ `6917d1d` (v0.3.2), `/v1/systemone` · 0.5 ms median latency | Apple M3 Pro (12-core CPU) | `max_state_bytes`: 2MB, `max_questions`: 64, `max_slots`: dynamic (all candidate criteria scored, nothing truncated) |

