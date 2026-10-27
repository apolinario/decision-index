# Model submissions

| Model | Edition | Decision Index | Complete results | Engine | Hardware |
|---|---|---:|---|---|---|
| [Drex Max v1.1](https://huggingface.co/NaceXDeepInfra/Drex-Max-v1.1) | 0.4 (public 0.3 suite) | 70.79 | [nace-ai/drex-max-v1.1-decision-index-results](https://huggingface.co/datasets/nace-ai/drex-max-v1.1-decision-index-results/tree/1c6b41a03b66d13f351b611c3193a0893b98bf11/runs/drex-max-v1.1) | in-process loader from the model repo (`code/drex_max`, `DecisionModel.predict`), kit `c254f8d` (edition 0.4) | NVIDIA B200 |

Inference settings: bf16, one forward pass per request with every question of the request, readout temperature from `decision_config.json`, max 65,536 tokens per request.

## Vision

| Model | Edition | Public vision index | Complete results | Engine | Hardware |
|---|---|---:|---|---|---|
| [Drex Max v1.1](https://huggingface.co/NaceXDeepInfra/Drex-Max-v1.1) | 0.4 (public 0.3 suite) | 75.78 | [nace-ai/drex-max-v1.1-decision-index-results](https://huggingface.co/datasets/nace-ai/drex-max-v1.1-decision-index-results/tree/1c6b41a03b66d13f351b611c3193a0893b98bf11/runs/drex-max-v1.1-vision) | in-process loader from the model repo (`code/drex_max`, `DecisionModel.predict`), kit `c254f8d` (edition 0.4) | NVIDIA B200 |

Vision settings: native resolution capped at 1,638,400 pixels per image, no author-set cap below that, no limit on images per request. The suite passes `vision verify`.
