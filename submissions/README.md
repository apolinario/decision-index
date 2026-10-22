# Submissions

One line per submitted run of the public suite (see [Submitting a model](../README.md#submitting-a-model-to-the-leaderboard)).

| Model | Hub repo | Results | Engine / kit commit | Hardware | Inference settings | Declared limits |
|---|---|---|---|---|---|---|
| gliclass-v4-edge | [knowledgator/gliclass-v4-edge](https://huggingface.co/knowledgator/gliclass-v4-edge) | [knowledgator/decision-index-results/runs/gliclass-v4-edge](https://huggingface.co/datasets/knowledgator/decision-index-results/tree/main/runs/gliclass-v4-edge) (edition 0.3, complete, public index 47.49) | `gliclass` engine (`decision_index/engines/gliclass_engine.py`, added in this PR; run as `gliclass_engine:GLiClassEngine`), gliclass 0.1.20, transformers 5.20.0, torch 2.12.1+cu130; kit `f5a6e4f` | 1x NVIDIA RTX PRO 6000 Blackwell Server Edition | uni-encoder, bf16, `max_length=16384`, `batch_size=8`, labels = option description, softmax over option logits | Inputs over 16,384 tokens refused as unsupported, never truncated (2 of 140,178 requests, both ToolRet) |
