# Submissions

One row per submission to the leaderboard, as described in the [README](../README.md#submitting-a-model-to-the-leaderboard).

| Model | Board | Model repo | Results | Engine and settings | Hardware |
|---|---|---|---|---|---|
| Joe0.1-27B | **vision** | [vantu-fit/Joe0.1-27B](https://huggingface.co/vantu-fit/Joe0.1-27B/tree/d35c74d5fa5e8736141550b1665226fe8d41e209) @ `d35c74d` | [runs/Joe0.1-27B/vision-scores.json](https://huggingface.co/datasets/vantu-fit/Joe0.1-27B-results/blob/b76a59dde56f78fceb7d3ee94aab06e15c285e76/runs/Joe0.1-27B/vision-scores.json) (complete, public vision index 76.64) | Custom engine `jev_vision_engine:JevVisionEngine` shipped in the model repo, driven through `decision-index vision pipeline` at commit `f5a6e4f`. One forward pass, letter read-out over the shown options, no generation. bfloat16, native resolution capped at 1,638,400 pixels per image, task `choice` with 2 to 26 options, nothing truncated, one request at a time. torch 2.8.0, transformers 5.20.0, flash-linear-attention 0.5.2. | 1x NVIDIA H200 141GB |

The run directory holds `results.jsonl` (gzipped), `environment.json`, `status.json` and `vision-scores.json`; `vision-scores.json` says `"complete": true` over all 12,210 rows with no errors. Weights, inference code and settings are in the model repo, which is public.
