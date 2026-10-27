# Model submissions

## Valen: updated text releases and vision results for board 0.4.1

These are complete public-only runs. Board release 0.4.1 uses the unchanged
public suite edition 0.3. Private evaluation and reference-GPU qualification
remain with the maintainers.

| Model and pinned weights | Track | Public index (0–100) | Complete scored run |
|---|---|---:|---|
| [Valen-0.8B](https://huggingface.co/Valen-Team/Valen-0.8B/tree/3471bde55167f8d0b16f70988e3a1f76efd79d60) | **text** | 20.79 | [scores.json](https://huggingface.co/datasets/Valen-Team/Valen-Decision-Index-Results/blob/ec0b6c7bb3ec29d5a4770486d98124df31c5c44e/releases/0.4.1/text/Valen-0.8B/scores.json) |
| [Valen-0.8B](https://huggingface.co/Valen-Team/Valen-0.8B/tree/3471bde55167f8d0b16f70988e3a1f76efd79d60) | **vision** | 47.76 | [vision-scores.json](https://huggingface.co/datasets/Valen-Team/Valen-Decision-Index-Results/blob/ec0b6c7bb3ec29d5a4770486d98124df31c5c44e/releases/0.4.1/vision/Valen-0.8B/vision-scores.json) |
| [Valen-2B](https://huggingface.co/Valen-Team/Valen-2B/tree/44b0a60511e5f1fbeb25d1b943b9d065006ea2f4) | **text** | 26.95 | [scores.json](https://huggingface.co/datasets/Valen-Team/Valen-Decision-Index-Results/blob/ec0b6c7bb3ec29d5a4770486d98124df31c5c44e/releases/0.4.1/text/Valen-2B/scores.json) |
| [Valen-2B](https://huggingface.co/Valen-Team/Valen-2B/tree/44b0a60511e5f1fbeb25d1b943b9d065006ea2f4) | **vision** | 56.44 | [vision-scores.json](https://huggingface.co/datasets/Valen-Team/Valen-Decision-Index-Results/blob/ec0b6c7bb3ec29d5a4770486d98124df31c5c44e/releases/0.4.1/vision/Valen-2B/vision-scores.json) |
| [Valen-4B](https://huggingface.co/Valen-Team/Valen-4B/tree/a94ffc427babf5c8959584ec8cd0cdd701487d6e) | **text** | 38.68 | [scores.json](https://huggingface.co/datasets/Valen-Team/Valen-Decision-Index-Results/blob/ec0b6c7bb3ec29d5a4770486d98124df31c5c44e/releases/0.4.1/text/Valen-4B/scores.json) |
| [Valen-4B](https://huggingface.co/Valen-Team/Valen-4B/tree/a94ffc427babf5c8959584ec8cd0cdd701487d6e) | **vision** | 62.52 | [vision-scores.json](https://huggingface.co/datasets/Valen-Team/Valen-Decision-Index-Results/blob/ec0b6c7bb3ec29d5a4770486d98124df31c5c44e/releases/0.4.1/vision/Valen-4B/vision-scores.json) |

- [Results dataset and artifacts](https://huggingface.co/datasets/Valen-Team/Valen-Decision-Index-Results/tree/ec0b6c7bb3ec29d5a4770486d98124df31c5c44e/releases/0.4.1); [manifest and fingerprints](https://huggingface.co/datasets/Valen-Team/Valen-Decision-Index-Results/blob/ec0b6c7bb3ec29d5a4770486d98124df31c5c44e/releases/0.4.1/manifest.json).
- [Exact frozen inference source, portable engine and replay commands](https://huggingface.co/datasets/Valen-Team/Valen-Decision-Index-Results/blob/ec0b6c7bb3ec29d5a4770486d98124df31c5c44e/releases/0.4.1/README.md#reproduce).
  The native adapter is `jevalkit.models.valen:ValenModel`; the portable upstream
  wrapper is `valen_engine:ValenEngine`. Weights and inference files match the
  pinned model revisions linked above.
- Suite and scorers: [`25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d`](https://github.com/apolinario/decision-index/tree/25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d).
  Each exported file was independently rescored; every benchmark score matches
  the original evaluation. Native answers, probabilities and statuses are retained.
- Hardware: NVIDIA H200, eight independent single-GPU workers per model/track,
  serial requests inside each worker. This is local latency, subject to the
  maintainers' RTX PRO 6000 measurement.
- Shared-state execution, SDPA, temperature 1.0, published native parameter dtypes
  and 32,768-token capacity. No calibration, truncation or option filtering.
- Text: 140,178 requests per model; one BRIGHT request has 32,891 compiled tokens
  and is unsupported (32 questions), retained in scoring. No runtime errors.
- **Vision**: all 11 benchmarks, 12,210 requests per model, 100% answer coverage,
  no runtime errors. Original images pass through the native processor capped
  at 1,638,400 pixels per image with the lower author cap lifted. Typed questions
  and image order are preserved. Environment files include the native cap and
  real source-item grid probes; full-run per-row grids were not logged.
