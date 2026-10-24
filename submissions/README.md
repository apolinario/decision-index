# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [Turion Jev 31B](https://huggingface.co/turion-ai/turion-jev-31b/tree/9edd5783ed1735cbf9e3e8ae7a546126ae05b0ce) by Turion AI (Gemma 4 31B, Apache-2.0) | 0.3 | 66.16 | [scores.json](https://huggingface.co/datasets/turion-ai/turion-jev-31b-decision-index-results/blob/cd7b034e9d249ea5556565286234b1393de2fb0c/runs/turion-jev-31b/scores.json) | `turion_jev.kit_engine:TurionJev` from the model repo's `code/` (Apache-2.0), in-process vLLM 0.32.0, online FP8; kit `25c6ec7` (see [turion-jev-31b](turion-jev-31b/README.md)) | 1× NVIDIA RTX PRO 6000 Blackwell Workstation Edition, one request at a time | 65,536 prompt tokens and 255 options per question; longer requests refused, never truncated |
