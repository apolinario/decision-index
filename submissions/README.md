# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [CoCo-Decision-4B](https://huggingface.co/corners-ai/CoCo-Decision-4B/tree/36a981d44065346b2732e057212bb5c3fc9c5029) | 0.3 | 47.80 | [scores.json](https://huggingface.co/datasets/corners-ai/CoCo-Decision-4B-decision-index/blob/64be2c17072cfd853f6f59cdc4448601cd921396/runs/CoCo-Decision-4B/scores.json) | kit `http` engine → `omj serve` ([iamupd/oh-my-jev@f0b8c78](https://github.com/iamupd/oh-my-jev/commit/f0b8c78684b05b18137ec8d1b797dfdcd7463a7a)), semif backend; kit `62d2f51` | 1× NVIDIA GeForce RTX 5090; bf16, one forward pass per question, one request at a time | 32,768 prompt tokens and 237 options per question; nothing in the suite exceeds either (0 unsupported) |
