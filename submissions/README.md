# Submissions

| Model | Decision Index | Results | Engine and exact code | Hardware | Declared capacity limits |
|---|---:|---|---|---|---|
| OneShot 27b (Qwen3.8-27B; [OneShotOrg/OneShot](https://huggingface.co/OneShotOrg/OneShot/tree/db7e5a20c7f23067d6c3bb9ea8beac8b0bf1828c) @ `db7e5a20c7f23067d6c3bb9ea8beac8b0bf1828c`) | **65.41** (0.3) | [scores.json](https://huggingface.co/datasets/OneShotOrg/OneShot_decision_index_results/blob/main/runs/oneshot/scores.json) | `http` (`POST /v1/systemone`, model `jev-policy`); BF16; Torch SDPA Flash + FLA; see the HF repository's startup script and deployment instructions for details | NVIDIA H20 | Input context: 65,535 tokens |

Checkpoint, server code and deployment instructions: [OneShotOrg/OneShot](https://huggingface.co/OneShotOrg/OneShot/tree/db7e5a20c7f23067d6c3bb9ea8beac8b0bf1828c) (private). The maintainers can deploy the checkpoint locally and evaluate it with the kit's built-in `http` engine.