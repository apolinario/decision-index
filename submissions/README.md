# Model submissions

| Model | Model repository | Complete public results | Engine / hardware / settings |
|---|---|---|---|
| Nemotron Deciffusion 8B v2 | [nemotron-deciffusion/Nemotron-Deciffusion-8B-v2](https://huggingface.co/nemotron-deciffusion/Nemotron-Deciffusion-8B-v2/tree/1dea38015e86e101eb978a33fa493818fc71e366) | [Decision Index 0.3 scores.json: 53.89](https://huggingface.co/datasets/nemotron-deciffusion/decision-index-0.3-results/blob/08c83d107f2bc8aa9f4f12847d4d9aae4b211c84/runs/nemotron-deciffusion-8b-v2/scores.json) | `http` engine against the released `/v1/systemone` container `ghcr.io/pst2154/nemotron-deciffusion:8b-v2` (FP8); kit 9eb2dbe2a358004c8782c66e40a83ac07b953fec; 4 x NVIDIA GB200, one server per GPU; [settings and reproduction](nemotron-deciffusion-8b-v2/README.md). |
