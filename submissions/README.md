# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [AutoTrust-JEV-27B v1.1](https://huggingface.co/autotrust/AutoTrust-JEV-27B-v1.1/tree/ae156900e9f338e423042b89e25d0fd8093064d4) (`autotrust/AutoTrust-JEV-27B-v1.1` @ `ae15690`) | 0.3 | 67.32 | [scores.json](https://huggingface.co/datasets/autotrust/jev-decision-index-results/blob/7fe2f14c4bc30bdd77e99a92dd9b63f2981b607b/runs/autotrust-jev-27b-v1.1/scores.json) | `jev_vllm_engine:JevVllmEngine` (shipped with the model) on vLLM 0.31.0 + StartLux prompt-bidirectional plugin, `serve_vllm.sh` ([submissions/autotrust-jev-27b-v1.1](autotrust-jev-27b-v1.1/README.md)); kit `f5a6e4f` | 1 × NVIDIA B200 | Text only; choice and noul; up to 255 options per question in one pass; context 65,536 tokens; no truncation; 0 unsupported |
