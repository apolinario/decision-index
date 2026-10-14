# Submissions

| Model | Track | Public index | Results | Engine and settings | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [Amos 1 large](https://huggingface.co/mukaddamzaid/amos-1-large/tree/362f4dc) | **vision** | **76.04** | [vision-scores.json](https://huggingface.co/datasets/mukaddamzaid/amos-1-decision-index-results/blob/9c59efcf7cc145e6935a2301f7bbc362fd4da0ca/runs/amos-1-large-vision/vision-scores.json) | the kit's `decide` engine against `amos.serve` `/v1/systemone` (the model repo's `serve/`), `--rotations 1`, label probabilities of one forward pass; kit `f8552a1` ([how to serve](Amos-1-large-vision/README.md)) | 1x H100 80GB, one request at a time | 131,072 tokens per request; 12,201 of 12,210 rows `ok`, 9 MMMU-Pro rows over the server's 4 MB body limit scored wrong; images capped at 1.6 MP (kit LANCZOS downscale), recorded in `environment.json` |
