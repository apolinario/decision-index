# Model submissions

| Model | Model repository | Complete public results | Engine / hardware / settings |
|---|---|---|---|
| Aplomb 2 Large | [empiriolabsai/aplomb-2-large](https://huggingface.co/empiriolabsai/aplomb-2-large/tree/ed20392ad09146884c82a21e938fe101d444516b) | [Decision Index 0.3 scores.json](https://huggingface.co/datasets/empiriolabsai/aplomb-2-large-decision-index/blob/f8a74734a8d4fe899cd7c6282e41a2ed8c7f4932/runs/aplomb-2-large/scores.json) | In-process engine `aplomb2_large.engine:LargeEngine` (`engine/` in the model repository), option `max_work=20000`; kit `25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d`; 8 x RTX PRO 6000 96GB (scored run, 8 shards) / 1 x RTX PRO 6000 96GB (latency); FP8 decoder projections, bf16 activations; [settings](aplomb-2-large/README.md). |
