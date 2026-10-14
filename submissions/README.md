# Submissions

| Model | Decision Index | Results | Engine and exact code | Hardware | Declared capacity limits |
|---|---:|---|---|---|---|
| Dinah-1 (`Lukitaduarte/dinah-1` @ `b518a5e`) | **34.32** | [full run and scores](https://huggingface.co/datasets/Lukitaduarte/dinah-1-decision-index-results/blob/99884c9911f2819ceced75c75e92820c48499a88/runs/dinah-1/scores.json) | `decision_index_engine:Dinah01Engine` (150M encoder, one pass per request, each option scored at its `[OPT]` marker, plus a code expert and a calculator expert chosen by the request content only; bfloat16) in the model repository (`decision_index_engine.py`, `dinah.py`, `experts.py`, `tools.py`) at `b518a5e`; kit `9eb2dbe` (0.3 scoring) | 1 x NVIDIA RTX PRO 5000, bf16 | Context 8,192 tokens, nothing truncated: 324 requests that do not fit are unsupported (8 scoreable). |
