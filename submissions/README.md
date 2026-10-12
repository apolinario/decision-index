# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [CoCo-Decision-4B 2.3.0](https://huggingface.co/corners-ai/CoCo-Decision-4B/tree/9996920c75fe16dcf5bdc0c24b4b57eb44ec3172) | 0.3 | 55.38 | [scores.json](https://huggingface.co/datasets/corners-ai/CoCo-Decision-4B-decision-index/blob/df4e585956f3b07583ee00f8e742a51c1f7e2dcd/runs/CoCo-Decision-4B-2.3.0/scores.json) | kit `http` engine → `omj serve` ([iamupd/oh-my-jev@292dc1b](https://github.com/iamupd/oh-my-jev/commit/292dc1bcb873d76b5ea7b3af3dfee8e0d22638cf)), semif backend, readout head over the option-letter tokens; kit `62d2f51` | 1× NVIDIA GeForce RTX 5090; bf16, one forward pass per question, one request at a time | 32,768 prompt tokens and 237 options per question; nothing in the suite exceeds either (0 unsupported) |
