# 𒉾𒅆𒁯𒉡 (peshigigunu)

Board: **vision** (Decision Index vision 0.3, public suite).

- Model: [plantwaterer/peshigigunu](https://huggingface.co/plantwaterer/peshigigunu/tree/dc8288666b50bc35bf6c5ff3d84af0dfc52a77b8) (pinned revision `dc8288666b50bc35bf6c5ff3d84af0dfc52a77b8`). The weights are a merge of two fine-tunes of Qwen/Qwen3.8-27B
  (Apache-2.0): [Rover00/WeMM-Decision](https://huggingface.co/Rover00/WeMM-Decision/tree/1543fbfcb171c7990607369e4c17329d3a7a6385)
  (Apache-2.0) plus half of the update of our plantwaterer/peshashaga, read through a fixed readout (the base model's language-model-head rows
  of the answer codes). The engine `peshigigunu.py` in that repository loads everything from the repository directory.
- **Prompt:** WeMM-Decision's "d3" request format (a system turn; the images, then State / Question / Options and a reply instruction),
  re-implemented in `peshigigunu.py`; it derives from d3 by the vLLM Semantic Router Team. No code from either is included.
- **A second model runs on every request:** PP-OCRv5 mobile (detection and English recognition, Apache-2.0, by the PaddlePaddle
  authors, through PaddleOCR) reads each image, and the text it finds is the prompt's state. The engine loads both OCR models from the
  Hub at pinned revisions and runs them on the same GPU. OCR time is inside every latency figure here.
- Results: [`runs/peshigigunu-vision-0.3`](https://huggingface.co/plantwaterer/peshigigunu/tree/dc8288666b50bc35bf6c5ff3d84af0dfc52a77b8/runs/peshigigunu-vision-0.3) in the same repository at the same revision
  (`results.jsonl` in the kit's compact form, `vision-scores.json`, `environment.json`, and each part's `environment` and `status` under `parts/`).
- Engine and options: `peshigigunu:Engine`, no options. `--workers 1 --compact`.
- Kit commit: `25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d`.
- Hardware: 1 x RTX PRO 6000 (96 GB) for the ten benchmarks other than Winoground, 1 x H200 for Winoground; bf16; readout in fp32.
- Resolution: every image is resized to between 589,824 and 1,638,400 pixels (aspect kept; smaller images are enlarged). The run's
  `effective_image_budget`: max_image_pixels 1,638,400, max_image_tokens 3,200. The OCR model reads each image at its
  original size, with its detector limited to 960 pixels on the longer side.
- Declared limits: choice questions only, one question per request, 2 to 12 options (the kit's letters A to L). No truncation of
  prompts; no cap on the number of images. One forward pass per request, nothing is decoded.
- Licence of the model repository: Apache-2.0 (WeMM-Decision's licence and NOTICE are carried in it). Part of the training data of
  peshashaga's fine-tunes is non-commercial or research-only, and three of its sources state no licence; WeMM-Decision's training data is
  not described by its authors. The model card lists them.
- Install: `bash install.sh` from the model repository: uv, Python 3.12 fetched by uv, everything pinned in `uv.lock`, then a self-check on
  the GPU; system packages `libgl1` and `libglib2.0-0`. PaddlePaddle's GPU wheel is not on PyPI; install.sh fetches it from PaddlePaddle's own storage, or from a mirror that holds an unmodified copy if that fails, and checks its sha256 either way.
  The environment is this repository's own: `bash install.sh` (`uv sync --frozen`, Python 3.12).

## What this run covers

The run has every row of the suite (12,210, all answered); `vision-scores.json` says `"complete": true`. It was made in two parts with the same files, engine, install and settings: Winoground (800 rows) on 1 x H200, the other
ten benchmarks (11,410 rows) on 1 x RTX PRO 6000. bf16 answers
and OCR text are not identical across GPU types, so a rerun on one GPU type will differ on some rows. The runs were made before the model had its name; the name was written into the label fields of the result rows and of the
environment and status files afterwards, and nothing else in them was changed (see the run directory's README).

| | value |
|---|---|
| public vision index (kit scorer) | **84.02** |
| rows answered | 12,210 / 12,210 |
| index of the ten benchmarks other than Winoground | 83.45 |

| benchmark | rows | accuracy | skill | weight |
|---|---|---|---|---|
| CV-Bench | 2,038 | 92.30 | 87.29 | 1 |
| BLINK | 961 | 88.03 | 81.47 | 1 |
| RealWorldQA | 589 | 86.76 | 78.36 | 1 |
| CharXiv | 532 | 89.47 | 85.96 | 1 |
| InfographicVQA | 244 | 99.18 | 98.91 | 1 |
| Mind2Web | 973 | 96.20 | 94.93 | 1 |
| Winoground | 800 | 94.50 | 89.00 | 1 |
| KIE (CORD+FUNSD) | 2,465 | 98.09 | 96.86 | 1 |
| Moderation (Hateful Memes) | 1,213 | 83.43 | 66.86 | 1 |
| R-Bench-M | 665 | 57.74 | 49.29 | 0.5 |
| MMMU-Pro vision | 1,730 | 64.68 | 59.49 | 0.25 |

Latency, one request at a time, the runner's `total_wall_ms`, OCR included. The 11,410 rows on 1 x RTX PRO 6000: median 333 ms,
mean 400 ms, 80th percentile 537 ms, 95th percentile 925 ms, slowest 1,657 ms (475 rows over 1,000 ms).
Model loading (114 s, including a file check and a kernel warm-up) is not included. The 800 Winoground rows on 1 x H200: median 389 ms,
mean 394 ms, 80th percentile 402 ms.

## Reproducing

```bash
cd <model repository>
bash install.sh                            # uv sync --frozen (Python 3.12) and a GPU self-check
uv run --frozen python -m decision_index vision pipeline --dir <work>/vision --engine peshigigunu:Engine --workers 1 --compact --out <runs>/peshigigunu
```
