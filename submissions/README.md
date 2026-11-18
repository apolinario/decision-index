# Decision Index submissions

| Model | Suite | Public index | Hub model | Public results | Engine and kit | Hardware | Inference settings |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| Stealth Sheep 2B | Text public 0.3 | 60.70 | [Model, `f65bcf13262c`](https://huggingface.co/yabel/stealth-sheep-2b/tree/f65bcf13262c76938a2be25cfbb627a455997d0f) | [Results](https://huggingface.co/datasets/yabel/stealth-sheep-2b-di-results/blob/9b5dac77abde13ce0c9f928bfd85bfc84ec697f2/runs/stealth-sheep-2b-di-text-0.3/scores.json) | [DI `25c6ec7b6b5f`](https://github.com/apolinario/decision-index/commit/25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d), HTTP `/v1/systemone` | 8 × NVIDIA H100 80 GB, one replica per GPU | Batch size 8, permutation averaging disabled, noncausal full attention, temperature 1.0, no truncation |
