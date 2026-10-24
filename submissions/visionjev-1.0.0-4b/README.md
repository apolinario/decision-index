# VisionJEV 1.0.0 (4B)

Submission for **vision and text**, public suite edition **0.3**.

- Model: [VisionJEV 1.0.0 (4B)](https://huggingface.co/tinnel123/VisionJEV-4B/tree/7c275c21de533de838bcd58c139837b95e570dea). Model revision: `7c275c21de533de838bcd58c139837b95e570dea`.
- Results revision: [`2b60a7586f770883c9539df4419110edb7106663`](https://huggingface.co/datasets/tinnel123/VisionJEV-4B-results/tree/2b60a7586f770883c9539df4419110edb7106663). Both tracks use this same model.
- Vision score: [`vision-scores.json`](https://huggingface.co/datasets/tinnel123/VisionJEV-4B-results/blob/2b60a7586f770883c9539df4419110edb7106663/runs/visionjev-1.0.0-4b-vision/vision-scores.json), complete, 12,210/12,210.
- Text score: [`scores.json`](https://huggingface.co/datasets/tinnel123/VisionJEV-4B-results/blob/2b60a7586f770883c9539df4419110edb7106663/runs/visionjev-1.0.0-4b-text/scores.json), complete, 140,178/140,178.
- Reproduction: [native engine, dependency versions, commands and capacity limits](https://huggingface.co/datasets/tinnel123/VisionJEV-4B-results/blob/2b60a7586f770883c9539df4419110edb7106663/REPRODUCTION.txt).
- Model loading: [CUDA and MLU installation and usage](https://huggingface.co/tinnel123/VisionJEV-4B/blob/7c275c21de533de838bcd58c139837b95e570dea/INFERENCE.md), including the separate, not newly scored CUDA reference entry.

| Track | Public index | Completed requests | Warm mean (ms) | Mean executed layers |
|---|---:|---:|---:|---:|
| Vision 0.3 | 73.03369467702923 | 12,210 | 247.27594685617964 | 32 |
| Text 0.3 | 57.78 | 140,178 | 86.70113558053053 | 32 |

## Exact runtime

| Setting | Recorded value |
|---|---|
| Official scoring kit | [`f8552a13b643d39f61185f5951bb5fed16bedf51`](https://github.com/apolinario/decision-index/tree/f8552a13b643d39f61185f5951bb5fed16bedf51) |
| Scientific inference source | Native `flash_adapter.py` / `warm_bridge.py`, checkpoint `inference_entry.load(checkpoint, device)`; source file hashes and original per-track code are in [source-provenance.json](https://huggingface.co/datasets/tinnel123/VisionJEV-4B-results/blob/2b60a7586f770883c9539df4419110edb7106663/source-provenance.json) and `provenance_source/` |
| Hardware and parallelism | Eight MLU590-M9DG devices; eight independent workers shard the requests |
| Recorded runtime | Python 3.12.12; PyTorch `2.10.0+cpu` with `torch_mlu 1.32.2+torch2.10.0`; Transformers 5.17.0. The `+cpu` package suffix does not mean CPU inference. Native FLA causal convolution is used; an exact FLA package version and peak memory were not recorded |
| Precision / depth | BF16 native inference; original complete FP32 weight storage; all 32 layers |
| Temperature | `1.5539092221462472`, inherited, applied once |
| Images | RGB/native processor, aspect ratio preserved, maximum 1,638,400 pixels per image; processor minimum 65,536 pixels |
| Context / packing | `max_length=262144`, `token_batch_budget=32768` |
| Options / questions / images | Up to 255 options; no fixed question-count or image-count limit beyond the sequence limit |
| Over-length handling | Bisect question groups while retaining the whole state and images; no input truncation or option filtering |
| Other historical options | `enable_thinking=false`, `compact=false`, `resume=false` |
| Timing | First call followed by a second actual warm forward on every request; each call synchronized; model loading excluded |

The actual loader enforces the image cap above. The old outer runner did **not** measure the largest effective image/token budget: its original rank metadata contains `max_image_pixels=0` and `max_image_tokens=null`. Those values are preserved and explained in the [observed-capacity record](https://huggingface.co/datasets/tinnel123/VisionJEV-4B-results/blob/2b60a7586f770883c9539df4419110edb7106663/runs/visionjev-1.0.0-4b-vision/observed-capacity-and-coverage.json); they are missing tracking, not a measured zero-image workload. No replacement observed maximum is invented. Please account for this reporting gap during review.

The original MLU inference path is retained. A CUDA entry is supplied separately for maintainer use; this submission does not claim that the reported MLU run was reproduced on CUDA. Likewise, the path-portable evaluation wrapper is provided without a new model run. See [reproduction instructions](https://huggingface.co/datasets/tinnel123/VisionJEV-4B-results/blob/2b60a7586f770883c9539df4419110edb7106663/REPRODUCTION.txt) and the model's usage documentation.

Public predictions are a compact export: response values, run identifiers and payload/image hashes are retained; request payloads, raw input/output, private paths and platform controls are omitted. Original score JSON bytes are unchanged. Historical `compact=false` describes the original execution, not this publication export.

The checkpoint uses a native Qwen3.5 backbone with 32 layers and a separate 255-by-2560 decision readout. It retains the original whole-state/question/options joint renderer and per-question `<|fim_suffix|>` marker readout. Full-attention layers are noncausal; linear-attention layers retain their native behavior. This protocol must be preserved to reproduce the results.

All supplied options are scored. Probabilities use the inherited temperature **1.5539092221462472**, applied exactly once to valid-option logits. The released backbone and trained readout retain their FP32 storage; the reported inference uses BF16 computation. This release executes all 32 layers and has no early-exit policy or LoRA adapter.

## Provenance and timing

The model is the final checkpoint of a 9,150-record, 36-update full-parameter supervised continuation of [pplx-decider-flash-v1.2](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.2/tree/6d771ad7b5ecff5e93b8fc7b90d4c5ccc341954d); it is not a fusion. The [model card](https://huggingface.co/tinnel123/VisionJEV-4B/blob/7c275c21de533de838bcd58c139837b95e570dea/README.md) describes the training mixture and preserves upstream attribution. Historical evaluation exposure of the upstream model is not independently established.

Warm latency is the sum of the device-synchronized per-request warm calls divided by the number of requests. Each request had a first call and a second, fully executed warm call; predictions were not cached or reused. Model loading is excluded. These are submitter measurements on MLU hardware, not Decision Index's single-process RTX PRO 6000 latency certification. Original scorer timing fields are preserved: they include the runner's two-call wrapper and therefore differ from the separately reported warm-call statistics.

No private/Full score or standardized-hardware latency certification is claimed. Please run the private evaluation and latency check using the published weights and native inference implementation.
