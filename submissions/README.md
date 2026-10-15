# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [CoCo-Decision-2.4-4B 2.4.0](https://huggingface.co/corners-ai/CoCo-Decision-2.4-4B/tree/58e9b7445760834a3ad9c704985102b0b08418d9) | 0.3 | 55.19 | [scores.json](https://huggingface.co/datasets/corners-ai/CoCo-Decision-2.4-4B-decision-index/blob/318daa4c0815607ff76fafd22d4a3419d67e2993/runs/CoCo-Decision-2.4-4B-2.4.0/scores.json) | kit `http` engine → `omj serve` ([iamupd/oh-my-jev@292dc1b](https://github.com/iamupd/oh-my-jev/commit/292dc1bcb873d76b5ea7b3af3dfee8e0d22638cf)), semif backend, readout head over the option-letter tokens; kit `62d2f51` | 1× NVIDIA GeForce RTX 5090; bf16, one forward pass per question, one request at a time | 32,768 prompt tokens and 237 options per question; nothing in the suite exceeds either (0 unsupported) |
