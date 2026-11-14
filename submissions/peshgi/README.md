# 𒉾𒄑𒄀 (peshgi)

Board: **vision** (Decision Index vision 0.3, public suite).

- Model: [plantwaterer/peshgi](https://huggingface.co/plantwaterer/peshgi/tree/6e8915481b58016d3b0186eda634801d85846a72) (pinned revision `6e8915481b58016d3b0186eda634801d85846a72`). An update to Qwen/Qwen3.8-27B (Apache-2.0) plus a 26-way
  answer head; the engine `peshgi.py` in that repository loads the base model from the Hub at a pinned revision.
- **A second model runs on every request:** PP-OCRv5 mobile (detection and English recognition, Apache-2.0, by the PaddlePaddle
  authors, through PaddleOCR) reads each image, and the text it finds is put in the prompt before the question. The engine loads
  both OCR models from the Hub at pinned revisions and runs them on the same GPU. OCR time is inside every latency figure here.
- Results: [`runs/peshgi-vision-0.3`](https://huggingface.co/plantwaterer/peshgi/tree/6e8915481b58016d3b0186eda634801d85846a72/runs/peshgi-vision-0.3) in the same repository at the same revision
  (`results.jsonl` in the kit's compact form, `vision-scores.json`, `environment.json`, and each part's `environment` and `status` under `parts/`).
- Engine and options: `peshgi:Engine`, no options. `--workers 1 --compact`.
- Kit commit: `25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d`.
- Hardware: 1 x RTX PRO 6000 (96 GB) for the ten benchmarks other than Winoground, 1 x H200 for Winoground; bf16; head in fp32.
- Resolution: native, up to 1,638,400 pixels per image (the base processor's default cap is lifted to that value). The run's
  `effective_image_budget`: max_image_pixels 1,638,400, max_image_tokens 3,200. The OCR model reads each image at its
  original size, with its detector limited to 960 pixels on the longer side.
- Declared limits: choice questions only, one question per request, 2 to 12 options (the kit's letters A to L). No truncation of
  prompts; no cap on the number of images. One forward pass per request, nothing is decoded.
- Licence of the model repository: Apache-2.0. Part of the training data is non-commercial or research-only, and three of its sources state no licence; the model card lists them.
- Install: `bash install.sh` from the model repository: uv, Python 3.12 fetched by uv, everything pinned in `uv.lock`, then a self-check on
  the GPU; system packages `libgl1` and `libglib2.0-0`. PaddlePaddle's GPU wheel is not on PyPI; install.sh fetches it from PaddlePaddle's own storage, or from a mirror that holds an unmodified copy if that fails, and checks its sha256 either way.
  The environment is this repository's own: `bash install.sh` (`uv sync --frozen`, Python 3.12).

## What this run covers

The run has every row of the suite (12,210, all answered); `vision-scores.json` says `"complete": true`. It was made in two
parts with the same files, engine, install and settings: Winoground (800 rows) on 1 x H200, the other ten benchmarks (11,410 rows)
on 1 x RTX PRO 6000. bf16 answers and OCR text are not identical across GPU types, so a rerun on one GPU type will differ on some
rows. The runs were made before the model had its name; the name was written into the label fields of the result rows and of the
environment and status files afterwards, and nothing else in them was changed (see the run directory's README).

| | value |
|---|---|
| public vision index (kit scorer) | **80.21** |
| rows answered | 12,210 / 12,210 |
| index of the ten benchmarks other than Winoground | 79.72 |

| benchmark | rows | accuracy | skill | weight |
|---|---|---|---|---|
| CV-Bench | 2,038 | 91.22 | 85.51 | 1 |
| BLINK | 961 | 87.41 | 80.51 | 1 |
| RealWorldQA | 589 | 82.00 | 70.59 | 1 |
| CharXiv | 532 | 86.28 | 81.70 | 1 |
| InfographicVQA | 244 | 99.59 | 99.45 | 1 |
| Mind2Web | 973 | 95.89 | 94.52 | 1 |
| Winoground | 800 | 92.25 | 84.50 | 1 |
| KIE (CORD+FUNSD) | 2,465 | 98.26 | 97.13 | 1 |
| Moderation (Hateful Memes) | 1,213 | 80.13 | 60.26 | 1 |
| R-Bench-M | 665 | 42.71 | 31.24 | 0.5 |
| MMMU-Pro vision | 1,730 | 55.43 | 48.88 | 0.25 |

Latency, one request at a time, the runner's `total_wall_ms`, OCR included. The 11,410 rows on 1 x RTX PRO 6000: median 323 ms,
mean 365 ms, 80th percentile 521 ms, 95th percentile 902 ms, slowest 1,637 ms (436 rows over 1,000 ms).
Model loading (120 s, including a kernel warm-up) is not included. The 800 Winoground rows on 1 x H200: median 373 ms,
mean 383 ms, 80th percentile 394 ms.

## Reproducing

```bash
cd <model repository>
bash install.sh                            # uv sync --frozen (Python 3.12) and a GPU self-check
uv run --frozen python -m decision_index vision pipeline --dir <work>/vision --engine peshgi:Engine --workers 1 --compact --out <runs>/peshgi
```
