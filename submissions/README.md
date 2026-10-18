# Submissions

| Model | Decision Index | Results | Engine and exact code | Hardware | Declared capacity limits |
|---|---:|---|---|---|---|
| Nimble-v4-max (`bespokelabs/Nimble-v4-max` @ `655aa705`, full fine-tune of `Qwen/Qwen3.8-27B` @ `1d4bf0f2`, bf16) | **68.46** | [full run and scores](https://huggingface.co/datasets/bespokelabs/nimble-v4-max-decision-index/blob/adea14289afa29dca3ff38b1fcb76e58f6add230/runs/nimble-v4-max/scores.json) | `nimble.evaluation.decision_index_engine:NimbleEngine` (in the results dataset under `code/`), kit `87d4650` runner (unchanged in `f5a6e4f`), scored as edition 0.3; options `shared_prefix=true`, `max_tails=48` | 32 x NVIDIA H200 (141 GB), one runner process per GPU | Choice questions up to 255 options; context 32,768 tokens. Nothing was truncated and no options were removed. All 152,955 requests returned `ok`. |
