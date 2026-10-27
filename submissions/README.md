# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [Aleria Judge 1.1.2](https://github.com/ALERIA-ML/aleria-judge/tree/afcb86f) (`google/gemma-4-31B-it` @ `842da37`, BF16, stock weights; no training) | 0.3 | 64.66 | [scores.json](https://huggingface.co/datasets/Aleria-AI/aleria-judge-1.1.2-decision-index/blob/e14883e76140bb356d85a240d1560015a7fa0425/runs/aleria-judge-1.1.2-text/scores.json) | the kit's `http` engine (`model=gemma-4-31b-trt`, `--compact`) against ALERIA-ML/aleria-judge@`afcb86f` (tag `v1.1.2`), `docker compose up -d` with every default (lean engine, effort `xhigh`) | 1× RTX PRO 6000 Blackwell Server Edition 96 GB, one request at a time | 1–255 options per question, 2–10 Score levels, 131,072 tokens per branch; never truncates (input written once when twice would not fit) |
