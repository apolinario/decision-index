# Submissions

| Model | Model repo | Results | Engine / commit | Hardware | Settings |
|---|---|---|---|---|---|
| jev-vision-27b | [e12ex2/jev-vision-27b](https://huggingface.co/e12ex2/jev-vision-27b) | [runs/jev-vision-27b](https://huggingface.co/datasets/e12ex2/jev-vision-27b-results/tree/main/runs/jev-vision-27b) | `http` engine, kit 9eb2dbe, against the repo's `code/serve/server.py` (`/v1/systemone`) | 1× H100 80GB (latency probe: 1× A100 80GB) | LoRA merged into Qwen/Qwen3.8-27B, bf16, vLLM, prefix caching, max-model-len 65536, 1 option order, letter-logit readout over listed options; declared limit 62 options per choice |
