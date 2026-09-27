# Submissions

Decision Index 0.2.1 (the 0.8B, 4B and 9B runs are complete 0.2 runs, rescored under 0.2.1 as the kit allows).

| Model | Decision Index 0.2.1 | Results | Engine and exact code | Hardware | Declared capacity limits |
|---|---:|---|---|---|---|
| JPT-35B-A3B (`kirp/jpt-35b-a3b`) | **52.89** | [full run and scores](https://huggingface.co/datasets/kirp/decision-index-results-jpt-35b-a3b/blob/main/scores.json) | [llm2jev](https://github.com/tic-top/llm2jev) 0.6.1 (chat prompt, thinking off) over SGLang 0.5.18 (tensor parallel 4); kit `87d4650` (0.2.1) | 8 x NVIDIA A100 40GB, 2 SGLang replicas (TP 4) + llm2jev, rows partitioned across 32 runners, merged and deduplicated by `run_id` before scoring | Options up to 255 per question (single-token labels); context 32,768 tokens. Nothing truncated, no options removed. |
| JPT-9B (`kirp/jpt-9b`) | **46.89** (0.2: 42.73) | [full run and scores](https://huggingface.co/datasets/kirp/decision-index-results-jpt-9b/blob/main/scores.json) | [llm2jev](https://github.com/tic-top/llm2jev) 0.6.1 (chat prompt, thinking off) over SGLang 0.5.9; run with kit `19ad28e`, rescored with `87d4650` (0.2.1) | 8 x NVIDIA A100 40GB, 2-8 SGLang + llm2jev replicas run in parallel with rows partitioned across them, merged and deduplicated by `run_id` before scoring | Options up to 255 per question (single-token labels); context 32,768 tokens. Nothing truncated, no options removed. |
| JPT-4B (`kirp/jpt-4b`) | **43.04** (0.2: 39.54) | [full run and scores](https://huggingface.co/datasets/kirp/decision-index-results-jpt-4b/blob/main/scores.json) | [llm2jev](https://github.com/tic-top/llm2jev) 0.6.1 (chat prompt, thinking off) over SGLang 0.5.9; run with kit `19ad28e`, rescored with `87d4650` (0.2.1) | 8 x NVIDIA A100 40GB, 2-8 SGLang + llm2jev replicas run in parallel with rows partitioned across them, merged and deduplicated by `run_id` before scoring | Options up to 255 per question (single-token labels); context 32,768 tokens. Nothing truncated, no options removed. |
| JPT-0.8B (`kirp/jpt-0.8b`) | **19.22** (0.2: 17.07) | [full run and scores](https://huggingface.co/datasets/kirp/decision-index-results-jpt-0.8b/blob/main/scores.json) | [llm2jev](https://github.com/tic-top/llm2jev) 0.6.1 (chat prompt, thinking off) over SGLang 0.5.9; run with kit `19ad28e`, rescored with `87d4650` (0.2.1) | 8 x NVIDIA A100 40GB, 2-8 SGLang + llm2jev replicas run in parallel with rows partitioned across them, merged and deduplicated by `run_id` before scoring | Options up to 255 per question (single-token labels); context 32,768 tokens. Nothing truncated, no options removed. |

All four are the same recipe (`mix_train_env_v11`: LoRA r=16 on every language-model projection of
Qwen3.5-{0.8B, 4B, 9B, 35B-A3B}, merged into full weights, multi-class Brier loss over option labels, one prefill per
question, no reasoning tokens), run through the same harness. Weights, training data and license (CC BY-NC 4.0) are on
each model's card. Not affiliated with TypeSafe AI.

The results datasets are gated because the suite carries GPQA and HLE item text; access requests are approved
promptly.
