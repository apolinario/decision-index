# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [JEV-35B](https://huggingface.co/autotrust/JEV-35B/tree/9190ed39e546dfb6173312ac9ebfbecdd8a08dbf) (`autotrust/JEV-35B` @ `9190ed3`) | 0.3 | 59.73 | [scores.json](https://huggingface.co/datasets/autotrust/jev-decision-index-results/blob/ce5850ff12caf28089b2b70143a14e7f328ed26f/runs/jev-35b/scores.json) | `jev_engine:JevEngine` with the vLLM batched driver `fast_run.py` ([submissions/jev-35b](jev-35b/README.md)); kit `9eb2dbe` | 1 × NVIDIA B200 | Up to 256 options per choice question in one pass; context 262,144 tokens; no truncation; 0 unsupported |
