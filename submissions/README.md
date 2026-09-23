# Submissions

| Model | Decision Index | Results | Engine and exact code | Hardware | Declared capacity limits |
|---|---:|---|---|---|---|
| Surogate Rune 26B-A4B (`surogate/rune-26b-a4b-GGUF`, JD-Q6_K) | **57.24** | [full run and scores](https://huggingface.co/datasets/surogate/decision-index-results/blob/main/runs/rune-26b-a4b-jd-q6k/scores.json) | `decision_index_surogate_engine_v1:SurogateEngine` (in the results dataset under `runs/rune-26b-a4b-jd-q6k/harness/`); kit `52a6989`; [surogate](https://github.com/invergent-ai/surogate) `19ff7d38` (release v1.5.2) decisions endpoint | 8 x NVIDIA RTX 5090 (400 W cap), one surogate server per GPU; 32 HTTP workers | Choice questions up to 255 options; context 32,768 tokens. Nothing was truncated and no options were removed. One HLE request was refused by the endpoint's template-boundary guard; it is recorded as an error and scored wrong. |
