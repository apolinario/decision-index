# Submissions

| Model | Decision Index | Results | Engine and exact code | Hardware | Declared capacity limits |
|---|---:|---|---|---|---|
| Reflex 27B (`Qwen/Qwen3.8-27B-FP8`, two permutations) | **56.23** | [full run and scores](https://huggingface.co/datasets/Kshetrajna/decision-index-results/blob/main/runs/reflex-27b-p2-full-r8-p32/scores.json) | `decision_index.engines.reflex_http:ReflexHttpSystemOne`; kit `52a6989` plus the [published harness and serving manifest](https://github.com/kshetrajna12/reflex/tree/9fcdb1c9d9325b30867bd4d72adaaa1d795c4da7/scripts/decision_index) | 8 x NVIDIA H100, one SGLang/Reflex replica per GPU; 32 HTTP workers | Choice questions are limited to 26 options: 14,500 requests were reported unsupported and scored wrong. No input was truncated and no options were removed. |
