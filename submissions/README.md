# Model submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [Sieve-27B](https://huggingface.co/sthanika-ai/Sieve-27B/tree/2273e0955486c4c0b6f780820b3ec8ddc90c4a98) | 0.3 | 57.06 | [scores.json](https://huggingface.co/datasets/sthanika-ai/Sieve-27B-decision-index-results/blob/d410c05e5885d1e2f33cab35f612963f5d37f3a7/runs/Sieve-27B/scores.json) | [`sieve.decision_index_engine:SieveEngine`](https://github.com/sthanika-ai/Sieve/blob/359f788bce4780a30d2928edf44b904465fdac82/sieve/decision_index_engine.py), `359f788bce4780a30d2928edf44b904465fdac82`; kit `62d2f51` (0.3) | 0.2.1 run on 2× NVIDIA A100 80GB PCIe (one half each), 0.3 GSM8K resume on 1× NVIDIA A100 80GB SXM; single process per GPU, bf16 eager, one request at a time | 65,536 state tokens; 32,768 question tokens; up to 255 choice options; no truncation (none reached: 0 unsupported) |
