# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [CoCo-Decision-4B 2.1.0](https://huggingface.co/corners-ai/CoCo-Decision-4B/tree/d17348beec75afbba9feb317163e7738a9c9439b) | 0.3 | 53.93 | [scores.json](https://huggingface.co/datasets/corners-ai/CoCo-Decision-4B-decision-index/blob/96c5635ccef0c986b451ef715e8ee34ebfb9efb0/runs/CoCo-Decision-4B-2.1.0/scores.json) | kit `http` engine → `omj serve` ([iamupd/oh-my-jev@292dc1b](https://github.com/iamupd/oh-my-jev/commit/292dc1bcb873d76b5ea7b3af3dfee8e0d22638cf)), semif backend, readout head over the option-letter tokens; kit `62d2f51` | 1× NVIDIA GeForce RTX 5090; bf16, one forward pass per question, one request at a time | 32,768 prompt tokens and 237 options per question; nothing in the suite exceeds either (0 unsupported) |
