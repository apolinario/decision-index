# Submissions

One line per submitted model.

| Model | Model repo | Results | Engine | Hardware | Settings | Public index |
|---|---|---|---|---|---|---|
| Drex v2.0 | [nace-ai/drex-v2.0](https://huggingface.co/nace-ai/drex-v2.0) (weights at `d911130ef6795cd4659302cf9b81e9f971c0e592`) | [nace-ai/drex-v2.0-decision-index-results](https://huggingface.co/datasets/nace-ai/drex-v2.0-decision-index-results) (`runs/drex-v2.0/scores.json`) | SGLang 0.5.21 with the `nace-drex-sglang` plugin, CUDA graphs on, behind a router, served over `/v1/systemone`, driven with the kit's `http` engine | NVIDIA B200 (Modal) | 131,072 token context, no sampling options. States up to 12,288 tokens are sent whole; longer ones are cut into 4K pieces with sparse selection and a relevance filter | 61.25 |

Results are for the public Decision Index 0.3 suite.
