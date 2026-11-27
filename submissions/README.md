# Submissions

One row per submitted model. Each model has its own folder with the details.

| Model | HF weights / revision | Public results | Inference code / commit | Hardware and settings |
|---|---|---|---|---|
| Nexus-SmallDecision-26B-A4B | [nexus-syntegra/Nexus-SmallDecision-26B-A4B-v2.0-SystemOne-Jev](https://huggingface.co/nexus-syntegra/Nexus-SmallDecision-26B-A4B-v2.0-SystemOne-Jev), weights pinned by `SHA256SUMS` in the repo (a single squashed commit); base `nvidia/Gemma-4-26B-A4B-NVFP4` @ `a19cfe00` | [Decision Index 0.3 run @ 2cae6192](https://huggingface.co/datasets/nexus-syntegra/nexus-smalldecision-26b-a4b-decision-index/tree/2cae619290e5b7213992dfa642d90f8dfc0fa7f0/runs/nexus-smalldecision-26b-a4b-v2.0) (index 53.54) | Kit `25c6ec7`, `http` engine. The run was served by the training repository's own runtime, which is unpublished. [`nexus_systemone`](Nexus-SmallDecision-26B-A4B/nexus_systemone) in this folder is a standalone port; the agreement between the two is measured in the details | 1 x RTX 5090 (32 GB), vLLM 0.30.0, read twice, one request at a time. [Details](Nexus-SmallDecision-26B-A4B/README.md) |
