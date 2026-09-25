# Submissions

| Model | Decision Index | Results | Engine and exact code | Hardware | Declared capacity limits |
|---|---:|---|---|---|---|
| JPT-0.8B (`kirp/jpt-0.8b`) | **17.07** | [full run and scores](https://huggingface.co/datasets/kirp/decision-index-results-jpt-0.8b/blob/main/scores.json) | [llm2jev](https://github.com/tic-top/llm2jev) 0.6.1 (chat prompt, thinking off) over SGLang 0.5.9; kit `19ad28e` | 8 x NVIDIA A100 40GB, 2-8 SGLang + llm2jev replicas run in parallel with rows partitioned across them, merged and deduplicated by `run_id` before scoring | Options up to 255 per question (single-token labels); context 32,768 tokens. Nothing truncated, no options removed. |
| JPT-4B (`kirp/jpt-4b`) | **40.33** | [full run and scores](https://huggingface.co/datasets/kirp/decision-index-results-jpt-4b/blob/main/scores.json) | [llm2jev](https://github.com/tic-top/llm2jev) 0.6.1 (chat prompt, thinking off) over SGLang 0.5.9; kit `19ad28e` | 8 x NVIDIA A100 40GB, 2-8 SGLang + llm2jev replicas run in parallel with rows partitioned across them, merged and deduplicated by `run_id` before scoring | Options up to 255 per question (single-token labels); context 32,768 tokens. Nothing truncated, no options removed. |
| JPT-9B (`kirp/jpt-9b`) | **42.73** | [full run and scores](https://huggingface.co/datasets/kirp/decision-index-results-jpt-9b/blob/main/scores.json) | [llm2jev](https://github.com/tic-top/llm2jev) 0.6.1 (chat prompt, thinking off) over SGLang 0.5.9; kit `19ad28e` | 8 x NVIDIA A100 40GB, 2-8 SGLang + llm2jev replicas run in parallel with rows partitioned across them, merged and deduplicated by `run_id` before scoring | Options up to 255 per question (single-token labels); context 32,768 tokens. Nothing truncated, no options removed. |

All three are the same recipe (`mix_train_env_v11`: LoRA r=16 on every attention/DeltaNet/MLP projection of
Qwen3.5-{0.8B,4B,9B}, merged into full weights, multi-class Brier loss over option labels, one prefill per question,
no reasoning tokens) at three sizes, run through the same harness. Weights, training data and license (CC BY-NC 4.0)
are on each model's card. Not affiliated with TypeSafe AI.

The results datasets are gated because the suite carries GPQA and HLE item text; access requests are approved
promptly.
