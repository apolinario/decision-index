# Submissions

One line per submitted model.

| Model | Model repo | Results | Engine | Hardware | Settings | Public index |
|---|---|---|---|---|---|---|
| Drex v1.5 | [nace-ai/drex-v1.5](https://huggingface.co/nace-ai/drex-v1.5) (weights at `ebf23bf183d2edfd0fce0e5bce6b5cf595546e0b`) | [nace-ai/drex-v1.5-decision-index-results](https://huggingface.co/datasets/nace-ai/drex-v1.5-decision-index-results) (`runs/drex-v1.5/scores.json`) | Kev (code in the model repo under `code/kev`; served build is `nace-ai/kev` commit `c091988`), served over `/v1/systemone`, driven with the kit's `http` engine | NVIDIA B200 (Modal), FP8 | 131,072 token context, no sampling options, state and questions sent unchanged in one request each | 58.08 |

Results are for Decision Index 0.3.1. The text suite is the same as 0.3, so the scores are the same.
