# Leaderboard submission

| Model | Tracks | Model and inference | Public results | Engine and settings |
|---|---|---|---|---|
| VisionJEV 1.0.0 | Text + Vision (0.3) | [model + inference](https://huggingface.co/taoszhang/VisionJEV/tree/7144d4da8f55d3c362461ebd8afe07b8856658f5) | [Text](https://huggingface.co/datasets/taoszhang/VisionJEV/blob/d6096aa71624d2aa912e623b7794fd3555fdb5c2/runs/visionjev-v1.0.0-text/scores.json), [Vision](https://huggingface.co/datasets/taoszhang/VisionJEV/blob/d6096aa71624d2aa912e623b7794fd3555fdb5c2/runs/visionjev-v1.0.0-vision/vision-scores.json) | Native MLU public runs, 4 × 8 devices; BF16, noncausal full attention, last-token pooling, T=1.0097098674430374, 8192-token limit, no truncation; Text batch ceiling 16; Vision batch 1 and native images capped at 1,638,400 pixels. Runnable CUDA adapter: visionjev.index_engine:VisionJEVEngine; original run kit commits and source hashes are documented in the PR and result provenance. |
