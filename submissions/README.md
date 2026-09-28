# Submissions

| Model | Decision Index | Results | Engine and exact code | Hardware | Declared capacity limits |
|---|---:|---|---|---|---|
| Dinah-0 (`Lukitaduarte/dinah-0` @ `e6bfc33`) | **27.63** | [full run and scores](https://huggingface.co/datasets/Lukitaduarte/dinah-0-decision-index-results/blob/f1c4c6f50d7ff6721022aa0d531f2d6f36346f18/runs/dinah-0/scores.json) | `decision_index_engine:DinahEngine` (150M encoder, one pass per request, each option scored at its `[OPT]` marker; bfloat16) in the model repository (`decision_index_engine.py`, `dinah.py`) at `e6bfc33`; kit `87d4650` (0.2.1 scoring) | 1 x NVIDIA RTX PRO 4000 Blackwell, bf16 | Context 8,192 tokens, nothing truncated: 324 requests that do not fit are unsupported (8 scoreable). |
