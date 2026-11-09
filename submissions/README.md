# Model submissions

| Board | Model | Organization | Weights and inference code | Complete public run | Public index |
| --- | --- | --- | --- | --- | ---: |
| vision | MJev-VL-4B | Mindlab | [checkpoint 002659](https://huggingface.co/mindlab-research/MJev-VL-4B/tree/0e22661351719711985a9ed4db45954c436571b8) | [Vision 0.3 scores](https://huggingface.co/datasets/mindlab-research/mjev-vl-4b-jev-index-decision-results/blob/8698fd461bbe62805b5a466f6539bd7bafed6802/runs/mjev-vl-4b-vision-merge-v1/vision-scores.json) | 70.67073552 |

## MJev-VL-4B

Mindlab requests private Vision evaluation of the revision above. It continues training from an earlier checkpoint on 680,571 visual decision fields. Public benchmarks informed model development. The [model card](https://huggingface.co/mindlab-research/MJev-VL-4B/blob/0e22661351719711985a9ed4db45954c436571b8/README.md) documents lineage, dependencies and overlap-audit limitations.

- **Public run.** All 12,210 requests across 11 benchmarks succeeded. The [compact results](https://huggingface.co/datasets/mindlab-research/mjev-vl-4b-jev-index-decision-results/blob/8698fd461bbe62805b5a466f6539bd7bafed6802/runs/mjev-vl-4b-vision-merge-v1/results.jsonl.gz) preserve the decisions, probabilities and payload hashes. The [rescore check](https://huggingface.co/datasets/mindlab-research/mjev-vl-4b-jev-index-decision-results/blob/8698fd461bbe62805b5a466f6539bd7bafed6802/runs/mjev-vl-4b-vision-merge-v1/rescore-verification.json) reproduces all fields of the original public score.
- **Engine.** [`general_mjev_decision_index_engine:GeneralMJevEngine`](https://huggingface.co/mindlab-research/MJev-VL-4B/blob/0e22661351719711985a9ed4db45954c436571b8/scripts/general_mjev_decision_index_engine.py), [Decision Index kit](https://github.com/apolinario/decision-index/tree/f5a6e4fd85fed05e77878dfc4966d0568bfb45ec). Native final-layer Q/K mean readout, BF16 decoder, FP32 readout, temperature 1. Each question is processed independently.
- **Limits.** Context 131,072 tokens, up to 8 ordered images, up to 255 options per question. The native processor caps each image at 1,638,400 pixels. The [run environment](https://huggingface.co/datasets/mindlab-research/mjev-vl-4b-jev-index-decision-results/blob/8698fd461bbe62805b5a466f6539bd7bafed6802/runs/mjev-vl-4b-vision-merge-v1/environment.json) records this same effective maximum.
- **Hardware.** Eight independent NVIDIA L20D shards, one GPU per shard. The standardized RTX PRO 6000 latency measurement is pending maintainer evaluation.

To reproduce, download the pinned model revision and use the runtime installation instructions in the model card. From the model directory:

```bash
python prepare_runtime.py --download-base
export PYTHONPATH="$PWD/scripts:$PWD/src:$PWD/vendor/decision-index"
python -m decision_index vision pipeline --dir /path/to/verified/vision-suite \
  --engine general_mjev_decision_index_engine:GeneralMJevEngine \
  --option checkpoint="$PWD" --option mode=vision --option device=cuda:0 \
  --option branch_budget=8192 --option max_branches=8 --out /path/to/new-run
```
