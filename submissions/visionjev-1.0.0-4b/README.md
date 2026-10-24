# VisionJEV 1.0.0 (4B)

Submission for **vision and text**, public suite **0.3**.

| Track | Public index | Completed requests | Warm mean (ms) | Mean executed layers |
|---|---:|---:|---:|---:|
| Vision 0.3 | 73.03369467702923 | 12,210 | 247.27594685617964 | 32 |
| Text 0.3 | 57.78 | 140,178 | 86.70113558053053 | 32 |

Both runs are complete with zero errors or missing requests. Text raw index: **67.71**.

- [Model and inference code](https://huggingface.co/tinnel123/VisionJEV-4B/tree/a5d1b24c42f546bdd8e05b34c28f46e86aff51e4) · [Loading instructions](https://huggingface.co/tinnel123/VisionJEV-4B/blob/a5d1b24c42f546bdd8e05b34c28f46e86aff51e4/INFERENCE.md)
- [Results](https://huggingface.co/datasets/tinnel123/VisionJEV-4B-results/tree/a57f283b0b98721395cd548498a508dd93241167) · [Vision scores](https://huggingface.co/datasets/tinnel123/VisionJEV-4B-results/blob/a57f283b0b98721395cd548498a508dd93241167/runs/visionjev-1.0.0-4b-vision/vision-scores.json) · [Text scores](https://huggingface.co/datasets/tinnel123/VisionJEV-4B-results/blob/a57f283b0b98721395cd548498a508dd93241167/runs/visionjev-1.0.0-4b-text/scores.json)
- [Reproduction commands](https://huggingface.co/datasets/tinnel123/VisionJEV-4B-results/blob/a57f283b0b98721395cd548498a508dd93241167/REPRODUCTION.txt) · [Inference source and hashes](https://huggingface.co/datasets/tinnel123/VisionJEV-4B-results/blob/a57f283b0b98721395cd548498a508dd93241167/source-provenance.json)

## Evaluation settings

| Setting | Recorded value |
|---|---|
| Scoring kit | [`f8552a13b643d39f61185f5951bb5fed16bedf51`](https://github.com/apolinario/decision-index/tree/f8552a13b643d39f61185f5951bb5fed16bedf51) |
| Inference entry | `inference_entry.load(checkpoint, device)`; native joint decision renderer and decision readout; per-track source hashes linked above |
| Hardware | Eight MLU590-M9DG devices, eight independent request workers |
| Runtime | Python 3.12.12; PyTorch `2.10.0+cpu` with `torch_mlu 1.32.2+torch2.10.0`; Transformers 5.17.0; native FLA causal convolution (exact version unrecorded) |
| Precision / depth | BF16 computation, FP32 checkpoint storage, all 32 layers |
| Temperature | `1.5539092221462472`, applied once to valid-option logits |
| Images | RGB, native processor, aspect ratio preserved; 65,536–1,638,400 pixels per image |
| Context / packing | `max_length=262144`; `token_batch_budget=32768` |
| Capacity | Up to 255 options; question and image counts limited by sequence length |
| Over-length handling | Split question groups while retaining the full state and images; no input truncation or option filtering |
| Other options | `enable_thinking=false`, `compact=false`, `resume=false`, `use_cache=false`, `fast_graphs=false` |

Warm mean measures the second synchronized forward per request, excluding model loading. Original scorer timing includes the two-call wrapper and is preserved separately. Private scores and maintainer RTX PRO 6000 latency are pending.

The historical runner did not record the observed maximum image/token budget (`0`/`null`); the loader's image cap is recorded above. A CUDA entry is included but has not rerun these MLU evaluations. See [capacity and reproduction limits](https://huggingface.co/datasets/tinnel123/VisionJEV-4B-results/blob/a57f283b0b98721395cd548498a508dd93241167/format-and-reproduction-limits.json).
