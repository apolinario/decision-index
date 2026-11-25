# Decision Index submissions

| Model | Suite | Public index | Hub model | Public results | Engine and kit | Hardware | Inference settings |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| Stealth Sheep 27B | Text public 0.3 | 73.30 | [Model](https://huggingface.co/yabel/stealth-sheep-27b/tree/16ad757463c6b03f6beebc9113d6c64510e0644d) | [Results](https://huggingface.co/datasets/yabel/stealth-sheep-27b-di-results/blob/6bb2b6344fe641ab8e922301aaf5c2f2450a3f5f/runs/stealth-sheep-27b-v2-di-text-0.3/scores.json) | [DI kit](https://github.com/apolinario/decision-index/commit/25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d), HTTP `/v1/systemone` | 8 × NVIDIA H100 80 GB, one replica per GPU | Batch size 1, permutation averaging disabled, noncausal full attention, temperature 1.0, FP32 readout, max 255 choices per question, no truncation |
