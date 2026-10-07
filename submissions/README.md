# Submissions

| Model | Decision Index | Results | Engine and exact code | Hardware | Declared capacity limits |
|---|---:|---|---|---|---|
| **DM-JEPA 1.5** (`DangerLabs/dm-jepa-1.5`) | **68.92** (**#1 Global Leaderboard**) | [scores and breakdown](https://huggingface.co/datasets/DangerLabs/decision-index-results/blob/0deed2f728e86b28580985b5c3cd2e172600dc12/runs/dm-jepa-1.5/scores.json) | Non-autoregressive System 1 JEPA 2.0 with Cartesian Block Butterfly layers, Riemannian Manifold Scorer, and `toks` assembly tokenizer | 1x NVIDIA RTX 3060 | Context 32,768 tokens; options up to 512 tokens. 0 unsupported requests. No truncation or filtering. Full valid distributions. |
| DM-JEPA 1.1 (`DangerLabs/DM-JEPA`) | **23.16** | [full run and scores](https://huggingface.co/datasets/DangerLabs/decision-index-results/blob/aacfb8dc266c75da3cbb13fc4c9bf4ca4c946c2a/runs/dm-jepa/scores.json) | Non-autoregressive JEPA System 1 architecture (`dm-jepa`); kit `87d4650` (0.2.1 scoring) | 1x NVIDIA GB10 (121.7GB) / NVIDIA RTX 3060 | Context 16,384 tokens; options up to 512 tokens. 241 scoreable requests unsupported. No truncation or option filtering. |
