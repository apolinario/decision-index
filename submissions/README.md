# Leaderboard submission

| Model | Tracks | Model and inference | Public results | Engine and settings |
|---|---|---|---|---|
| VisionJEV 1.0.0 | Text + Vision (0.3) | [model + inference](https://huggingface.co/taoszhang/VisionJEV/tree/cd6353aaea57561bd8c9beee69311f104dd77528) | [Text](https://huggingface.co/datasets/taoszhang/VisionJEV/blob/0f05503cb3dbd45ecb25a93bb2ba92fb7ae92495/runs/visionjev-v1.0.0-text/scores.json), [Vision](https://huggingface.co/datasets/taoszhang/VisionJEV/blob/0f05503cb3dbd45ecb25a93bb2ba92fb7ae92495/runs/visionjev-v1.0.0-vision/vision-scores.json) | Native MLU public runs, 4 × 8 devices; BF16, noncausal full attention, last-token pooling, T=0.986316004609301, 8192-token limit, no truncation; Text batch ceiling 16; Vision batch 1 and native images capped at 1,638,400 pixels. Runnable CUDA adapter: visionjev.index_engine:VisionJEVEngine; original run kit commits and source hashes are documented in the PR and result provenance. |
