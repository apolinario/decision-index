# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [Orion Compass](https://huggingface.co/SubstrateAILab/orion-compass/tree/46f8d15dd15d25c397a5f102c4f60598a94b95aa) (Gemma 4 31B, Apache-2.0) | 0.3 | 65.90 | [scores.json](https://huggingface.co/datasets/SubstrateAILab/orion-compass-decision-index/blob/55605b9f29c6de5d79a0c842ac391b9bd219738a/runs/orion-compass-flexdm/scores.json) | `jevlab.vllm_reader:VllmReader` from [FlexDM](https://github.com/1000luca/FlexDM/tree/d147cd7e) v1.0 @ `d147cd7e` (MIT), in-process vLLM 0.30.0, online `fp8_per_block`; kit `9eb2dbe` (see [orion-compass](orion-compass/README.md)) | 3× NVIDIA RTX PRO 6000 Blackwell Server Edition, one shard each, one request at a time | 131,072 tokens and 255 options per question; longer requests refused, never truncated |
