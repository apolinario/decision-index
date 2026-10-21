# Submissions

One row per submission to the leaderboard, as described in the [README](../README.md#submitting-a-model-to-the-leaderboard).

| Model | Board | Model repo | Results | Engine and settings | Hardware |
|---|---|---|---|---|---|
| RBDash-Decision-v0.1 | **vision** | [RBDash-Team/RBDash-Decision-v0.1](https://huggingface.co/RBDash-Team/RBDash-Decision-v0.1) | [runs/vision-rbdash-decision-v0.1/vision-scores.json](https://huggingface.co/datasets/RBDash-Team/decision-index-results/blob/main/runs/vision-rbdash-decision-v0.1/vision-scores.json) (complete, public vision index 73.49) | Custom engine `rbdash_decision_engine:RBDashEngine` from the model repo, through `decision-index vision pipeline` at commit `f5a6e4f`. Native classification: one forward pass, 255-way readout over the option codes, temperature 0.8821 from `decision_config.json`. bfloat16, native resolution capped at 1,638,400 pixels per image, one to four images, nothing truncated (inputs over 32,768 tokens raise an error), one request at a time. `flash-linear-attention` 0.5.2 and `causal-conv1d` 1.7.0 installed, torch 2.14.1, transformers 5.20.2. | 1x A800-SXM4-80GB |

The run directory holds `results.jsonl` (gzipped), `environment.json`, `status.json` and `vision-scores.json`; `vision-scores.json` says `"complete": true` over all 12,210 rows with no errors. Inference code, weights and the exact settings are in the model repo, which is public.
