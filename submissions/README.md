# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [CoCo-Decision-4B 2.0.0](https://huggingface.co/corners-ai/CoCo-Decision-4B/tree/949f1080f3aa58d61481e16456919c81b7b78207) | 0.3 | 51.93 | [scores.json](https://huggingface.co/datasets/corners-ai/CoCo-Decision-4B-decision-index/blob/bf016abc576707bc5ab8a192f7684afe4879298c/runs/CoCo-Decision-4B-2.0.0/scores.json) | kit `http` engine → `omj serve` ([iamupd/oh-my-jev@f0b8c78](https://github.com/iamupd/oh-my-jev/commit/f0b8c78684b05b18137ec8d1b797dfdcd7463a7a)), semif backend; kit `62d2f51` | 1× NVIDIA GeForce RTX 5090; bf16, one forward pass per question, one request at a time | 32,768 prompt tokens and 237 options per question; nothing in the suite exceeds either (0 unsupported) |
