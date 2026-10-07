# Submissions

| model | model repo | results | engine / code | hardware | settings |
|---|---|---|---|---|---|
| decider-31b | [Mapika/decider-31b](https://huggingface.co/Mapika/decider-31b) @ cf50c0b | [Mapika/decision-index-results/runs/decider-31b](https://huggingface.co/datasets/Mapika/decision-index-results/tree/main/runs/decider-31b) | kit 0.3 `http` engine (62d2f51) → `decider.serve_vllm`, decider-ai 1.9.0, vLLM 0.29.0 | b300 | `DECIDER_MODEL=Mapika/decider-31b uvicorn decider.serve_vllm:app`; everything else from the repo's `decider_config.json`; no truncation, no capacity limits |
