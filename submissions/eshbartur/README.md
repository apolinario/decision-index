# eshbartur

Decision Index 0.3 public score: **63.23** (self-run; balanced raw 72.56, breadth skill 61.99). 140,620 requests run, 140,178 scored (`complete: true`), 0 unsupported, 0 errors.

- Model: https://huggingface.co/plantwaterer/eshbartur/tree/bba7da590987ba722f3f38b12dd5762bcee25d32
- Results: https://huggingface.co/datasets/plantwaterer/eshbartur-decision-index/tree/31d3f0f1f94a3d4cf8b769b1a68d6fd3fc6cc676/runs/eshbartur
- Kit: `apolinario/decision-index` @ `9eb2dbe2a358004c8782c66e40a83ac07b953fec`.
- Engine: `eshbartur:Engine`, in-process, from the model repository, no engine options (`--compact`).

## Settings

```bash
uvx --from huggingface_hub hf download plantwaterer/eshbartur --local-dir eshbartur
cd eshbartur
bash install.sh
uv run --frozen python -m decision_index run --engine eshbartur:Engine --compact --out runs/eshbartur
```

- Pure-text `Qwen3_5ForCausalLM` (4.21B parameters, vision tower and MTP head removed from `Qwen/Qwen3.5-4B` revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`), bf16 weights (8.41 GB), 255-way fp32 linear readout with held-out calibration temperature `T = 1.40` folded into weight and bias.
- Non-causal full attention on the 8 full-attention layers (`AttentionMaskInterface`), causal on the 24 Gated DeltaNet layers, RE2 (Read-Twice) for prompts `<= 2,048` tokens, and right-padded batched scoring for multi-question requests. Native support for `choice`, `noul`, and `score` up to 255 options; `max_len = 65,536`, no truncation.
- Hardware: fits on one 16 GB+ GPU (peak VRAM 12.73 GB on RTX PRO 6000). Complete 140,620-row run executed in 8 shards on 8 x NVIDIA H200; single-GPU sequential latency measured on 1 x NVIDIA RTX PRO 6000 Blackwell 96 GB.
- Latency (1 x NVIDIA RTX PRO 6000 Blackwell, one request at a time, 1,000-row public board sample, cold kernel cache): **median 37.6 ms**, mean 229.5 ms, p80 92.4 ms, p95 360.8 ms, peak VRAM 12.73 GB (`99.91%` choice parity vs H200 run across 2,327 choices).
- Licence of the model and results repositories: Apache-2.0. Some training data sources carry non-commercial, research-only, or unstated licences; the model card notes them.
