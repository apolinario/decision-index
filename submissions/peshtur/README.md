# peshtur

Board: **vision** (Decision Index vision 0.3, public suite).

- Model: [plantwaterer/peshtur](https://huggingface.co/plantwaterer/peshtur/tree/73e6683622487fbc0aa8046afeb553b7d805bcaf) (pinned revision `73e6683622487fbc0aa8046afeb553b7d805bcaf`). Fine-tuned from `Qwen/Qwen3.5-4B` (revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, Apache-2.0) with rank-16 LoRA merged into 248 language-model linear matrices, vision encoder and vision merger frozen, and a learned 26-way fp32 linear readout (`readout.safetensors`: weight and bias over answer codes `A..Z`). The standalone engine `peshtur.py` loads everything from the repository directory.
- **Prompt:** the `d3` request format (system turn; images followed by State / Question / Options and reply instruction), re-implemented in `peshtur.py` (derives from d3 by the vLLM Semantic Router Team / Rover00/WeMM-Decision; no code from either is included).
- **A second model runs on every request:** PP-OCRv5 mobile (`PaddlePaddle/PP-OCRv5_mobile_det` + `PaddlePaddle/en_PP-OCRv5_mobile_rec`, Apache-2.0, through PaddleOCR) reads each image, and when text is found it is placed in the prompt's state. Both OCR models are loaded from the Hub at pinned revisions and run on the same GPU; live OCR time is included in all latency figures below.
- Results: [`runs/peshtur-vision-0.3`](https://huggingface.co/plantwaterer/peshtur/tree/73e6683622487fbc0aa8046afeb553b7d805bcaf/runs/peshtur-vision-0.3) in the same repository at the same revision (`results.jsonl` compact form, `vision-scores.json`, `environment.json`, `latency.json`, and `parts/`).
- Engine and options: `peshtur:Engine`, no options, `--workers 1 --compact`.
- Kit commit: `25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d`.
- Hardware: NVIDIA H200 for the complete 12,210-row scored run (the 10 non-Winoground benchmarks across 4 processes on 4 GPUs, Winoground in 1 process); 1 x NVIDIA RTX PRO 6000 Blackwell (96 GB) for single-GPU sequential latency on the 547-row sample; bf16 weights, fp32 readout.
- Resolution: every image resized to between 589,824 and 1,638,400 pixels (aspect ratio preserved; smaller images enlarged). Effective image budget: `max_image_pixels = 1,638,400`, `max_image_tokens = 3,200`. OCR reads each image at original size with detector side cap 960 px.
- Declared limits: choice questions only, one question per request, 2 to 26 options (`A..Z`). No prompt truncation; no image-count cap; single forward pass with no token decoding.
- Licence: Apache-2.0 for all weights and code in the model repository (`LICENSE` and `NOTICE` included). Some training data sources carry non-commercial, research-only, or unstated licences (detailed in the model card).
- Install: `bash install.sh` from the model repository (`uv sync --frozen`, Python 3.12, plus GPU `selfcheck.py`); requires `libgl1` and `libglib2.0-0` for OpenCV. Verified clean cold-cache install on NVIDIA RTX PRO 6000 Blackwell Server Edition.

## What this run covers

Complete run of all 12,210 rows of the Decision Index Vision 0.3 public suite, all answered (`"complete": true`, `0` errors), executed directly by `peshtur:Engine` (no relabelling).

| Metric | Value |
|---|---:|
| Public vision index (all 11 benchmarks, kit scorer) | **75.44** |
| Rows answered | 12,210 / 12,210 |
| Ten benchmarks excluding Winoground | 75.06 |

| Benchmark | Rows | Accuracy (%) | Skill (%) | Weight |
|---|---:|---:|---:|---:|
| CV-Bench | 2,038 | 90.19 | 83.81 | 1.00 |
| BLINK | 961 | 85.54 | 77.61 | 1.00 |
| RealWorldQA | 589 | 75.72 | 60.32 | 1.00 |
| CharXiv | 532 | 82.33 | 76.44 | 1.00 |
| InfographicVQA | 244 | 95.90 | 94.54 | 1.00 |
| Mind2Web | 973 | 93.83 | 91.78 | 1.00 |
| Winoground | 800 | 89.38 | 78.75 | 1.00 |
| KIE (CORD+FUNSD) | 2,465 | 97.48 | 95.86 | 1.00 |
| Moderation (Hateful Memes) | 1,213 | 79.14 | 58.29 | 1.00 |
| R-Bench-M | 665 | 33.38 | 20.05 | 0.50 |
| MMMU-Pro vision | 1,730 | 41.21 | 32.58 | 0.25 |

**Latency (1 x NVIDIA RTX PRO 6000 Blackwell Server Edition 96 GB, sequential, live OCR included):** measured over the fixed 547-row sample (`sha1(id) % 20 == 0` of the 10 non-Winoground benchmarks) starting from empty kernel caches, excluding 87.3 s load + warm-up: **median 154.3 ms**, mean 200.7 ms, p80 257.5 ms, p95 544.7 ms, p99 945.5 ms, max 1,210.6 ms (`3 / 547` over 1,000 ms; median OCR 40.2 ms; peak VRAM 12,531 MiB (13.1 GB)). Cross-GPU parity vs the H200 scored run on the 547 rows: `544 / 547` (`99.45%`) identical answers, `index_diff = 0.00`.

## Reproducing

```bash
cd <model repository>
bash install.sh
uv run --frozen python -m decision_index vision pipeline --dir <work>/vision --engine peshtur:Engine --workers 1 --compact --out <runs>/peshtur
```
