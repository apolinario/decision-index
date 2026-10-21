# Submissions

| Model | Results | Engine / commit | Hardware |
|---|---|---|---|
| Scion-1 9B (Qwen3.5-9B + LoRA r64, prompt repetition, temperature 1.7278) | [runs/scion-v4](https://huggingface.co/datasets/sinan-fireworks/decision-index-results/tree/main/runs/scion-v4) (gated: request access) | `decision_index.engines.scion:ScionEngine` (kit 87d4650 + `engines/scion.py`), options `base=Qwen/Qwen3.5-9B lora=sinan-fireworks/scion-1-9b-lora temperature=1.7278 repeat=true` | 1 x NVIDIA RTX PRO 6000 Blackwell Server Edition (Modal) |
