# dormice

Board: **vision** (Decision Index vision 0.3, public suite).

- Model: [plantwaterer/dormice](https://huggingface.co/plantwaterer/dormice/tree/13caf7d0503a14fe6e0b8d3570a76ea88acdcf48) (pinned revision `13caf7d0503a14fe6e0b8d3570a76ea88acdcf48`). An update to Qwen/Qwen3.8-27B (Apache-2.0) plus a 26-way
  answer head; the engine `dormice.py` in that repository loads the base model from the Hub at a pinned revision.
- Results: [`runs/dormice-vision-0.3`](https://huggingface.co/plantwaterer/dormice/tree/13caf7d0503a14fe6e0b8d3570a76ea88acdcf48/runs/dormice-vision-0.3) in the same repository at the same revision
  (`results.jsonl` in the kit's compact form, `vision-scores.json`, `environment.json`, and each part's `environment` and `status` under `parts/`).
- Engine and options: `dormice:Engine`, no options. `--workers 1 --compact`.
- Kit commit: `25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d`.
- Hardware: 1 x RTX PRO 6000 (96 GB) for the ten benchmarks other than Winoground, 1 x H200 for Winoground; bf16; head in fp32.
- Resolution: native, up to 1,638,400 pixels per image (the base processor's default cap is lifted to that value). The run's
  `effective_image_budget`: max_image_pixels 1,638,400, max_image_tokens 3,200.
- Declared limits: choice questions only, one question per request, 2 to 12 options (the kit's letters A to L). No truncation of
  prompts; no cap on the number of images. One forward pass per request, nothing is decoded.
- Licence of the model repository: Apache-2.0. Part of the training data is non-commercial or research-only; the model card lists those sources.

## What this run covers

The run has every row of the suite (12,210, all answered); `vision-scores.json` says `"complete": true`. It was made in two
parts with the same files, engine and settings: Winoground (800 rows) on 1 x H200, the other ten benchmarks (11,410 rows) on
1 x RTX PRO 6000. `results.jsonl` is the two parts' compact rows one after the other, unchanged. bf16 answers are not identical
across GPU types, so a rerun on one GPU type will differ on some rows; the model card gives what we measured.

| | value |
|---|---|
| public vision index (kit scorer) | **78.38** |
| rows answered | 12,210 / 12,210 |
| index of the ten benchmarks other than Winoground | 77.82 |

| benchmark | rows | accuracy | skill | weight |
|---|---|---|---|---|
| CV-Bench | 2,038 | 88.42 | 80.90 | 1 |
| BLINK | 961 | 85.54 | 77.61 | 1 |
| RealWorldQA | 589 | 82.51 | 71.42 | 1 |
| CharXiv | 532 | 85.71 | 80.95 | 1 |
| InfographicVQA | 244 | 99.18 | 98.91 | 1 |
| Mind2Web | 973 | 95.17 | 93.56 | 1 |
| Winoground | 800 | 91.62 | 83.25 | 1 |
| KIE (CORD+FUNSD) | 2,465 | 98.05 | 96.79 | 1 |
| Moderation (Hateful Memes) | 1,213 | 77.41 | 54.82 | 1 |
| R-Bench-M | 665 | 41.65 | 29.97 | 0.5 |
| MMMU-Pro vision | 1,730 | 51.04 | 43.85 | 0.25 |

Latency, one request at a time, the runner's `total_wall_ms`. The 11,410 rows on 1 x RTX PRO 6000: median 199 ms,
mean 290 ms, 80th percentile 423 ms, 95th percentile 820 ms, slowest 1,372 ms
(194 rows over 1,000 ms). The 800 Winoground rows on 1 x H200: median 342 ms, mean 350 ms, 80th percentile
355 ms, slowest 958 ms. Model loading (109 s, including a kernel warm-up) is not included.

## Reproducing

```bash
pip install -r requirements.txt            # from the model repository
PYTHONPATH=<model repository> python -m decision_index vision pipeline --dir work/vision \
    --engine dormice:Engine --workers 1 --compact --out runs/dormice
```
