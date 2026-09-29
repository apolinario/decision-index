# JEV-Gemma4-26B-A4B

| Model | Edition | Decision Index | Complete results | Engine | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [JEV-Gemma4-26B-A4B](https://huggingface.co/autotrust/JEV-Gemma4-26B-A4B/tree/603bfed48410726eac15681bc9c55da904547e2d) (`autotrust/JEV-Gemma4-26B-A4B` @ `603bfed`; LoRA + 24-slot decision head on `google/gemma-4-26B-A4B-it`) | 0.2.1 | 58.05 | [scores.json](https://huggingface.co/datasets/autotrust/jev-decision-index-results/blob/6c4044b2128772a30620adb4c75da615755f0f8d/runs/jev-gemma4-26b-a4b/scores.json) | `jev_engine:JevEngine` (this directory); batched driver `fast_run.py` (64 requests per batch) for all rows | 1 × NVIDIA B200 | Context 262,143 tokens; 16 choice options per pass, wider questions (up to 255) read in groups of ≤ 16 plus a final, no option removed; no truncation |

Training data includes the public training splits of 12 datasets whose test splits the suite uses; see the pull request.

```sh
# this directory on PYTHONPATH
python -m decision_index pipeline --engine jev_engine:JevEngine --option model=autotrust/JEV-Gemma4-26B-A4B \
  --option revision=603bfed48410726eac15681bc9c55da904547e2d --out runs/jev-gemma4-26b-a4b
```
