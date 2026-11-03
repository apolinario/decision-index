# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [Orion Compass](https://huggingface.co/SubstrateAILab/orion-compass/tree/47e965e015cde4934cd968e6fb73ded95a1b6a5c) (Gemma 4 31B, Apache-2.0) | 0.3 | 67.83 | [scores.json](https://huggingface.co/datasets/SubstrateAILab/orion-compass-decision-index/blob/a96e927a91e34dac357945d56258935ff2267603/runs/orion-compass-flexdm/scores.json) | `jevlab.vllm_reader:VllmReader` from [FlexDM](https://github.com/1000luca/FlexDM/tree/d147cd7e) v1.0 @ `d147cd7e` (MIT), in-process vLLM 0.30.0, online `fp8_per_block`; kit `25c6ec7` (see [orion-compass](orion-compass/README.md)) | 3× NVIDIA RTX PRO 6000 Blackwell Server Edition, one shard each, one request at a time | 131,072 tokens and 255 options per question; longer requests refused, never truncated |
