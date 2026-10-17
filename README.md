# Model submissions

| Model | Track | Public index | Results | Engine and settings | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [jev-d3-46k](jev-d3-46k-vision/README.md) | **vision** | **68.26486972026493** · 12,210/12,210 rows `ok` | [tinnel123/jev-d3-46k](https://huggingface.co/tinnel123/jev-d3-46k/tree/d0de482276e3506b10b53e04f971cd3df7e25817) · [results run](https://huggingface.co/datasets/tinnel123/decision-index-results/tree/5235be8201e123c7cf720e2d53e9ecfb5f38b7d4/runs/d3_full_checkpoint_vision11_68_2649) · [`vision-scores.json`](https://huggingface.co/datasets/tinnel123/decision-index-results/blob/5235be8201e123c7cf720e2d53e9ecfb5f38b7d4/runs/d3_full_checkpoint_vision11_68_2649/official_scores/vision-scores.json) | Custom standalone D3 full-checkpoint runtime, BF16 backbone + FP32 readout, one materialized-model forward; [engine code](https://huggingface.co/tinnel123/jev-d3-46k/blob/d0de482276e3506b10b53e04f971cd3df7e25817/d3_engine.py); kit `f5a6e4f` | 8 MLU accelerator cards; exact card model is not recorded in the public receipt | Native image preprocessing capped at 1,638,400 px (`effective_image_budget=1638400`); no truncation or option filtering; up to 255 options; one request at a time |

The public result is the public half of the Vision 0.3 Full score. The maintainers run the private half.
