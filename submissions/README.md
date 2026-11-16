# Model submissions

| Model | Model revision | Public results | Engine / kit | Hardware | Inference settings |
|---|---|---|---|---|---|
| [Sloth-27B v1](https://huggingface.co/cjqhhh/Sloth-27B-v1) | `505e8a98e1ed5ff28fbdd3a11d3b7fe0eb136f9d` | [scores.json](https://huggingface.co/datasets/cjqhhh/Sloth-27B-v1-results/resolve/e0c2ede894dafc56e07ef0d7abdc0ab83b5eca76/runs/sloth-27b-v1/scores.json), [reproduction](https://huggingface.co/datasets/cjqhhh/Sloth-27B-v1-results/blob/e0c2ede894dafc56e07ef0d7abdc0ab83b5eca76/REPRODUCE.md) | `d3_vllm_engine:D3VLLMEngine` at model revision; kit `25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d` | 8 independent H20 workers; TP=1 each | BF16; vLLM 0.33.0; Torch 2.14.0+cu130; Transformers 5.20.2; max length 32768; GPU memory 0.75; max sequences 24; batch invariant 0; temperature 1.0 |
