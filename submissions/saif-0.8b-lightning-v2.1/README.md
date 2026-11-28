# Saif 0.8B Lightning v2.1 (0.8B)

Saif 0.8B Lightning is GyrTechnology's small decision model, a full fine-tune of Qwen3.5-0.8B, trained on a mix of data in two stages with a WSD schedule. v2.1 is a weight average with v2. It returns one probability for every option of every question in a single forward pass and does not generate text.

| | Saif 0.8B Lightning v2.1 |
|---|---:|
| Decision Index 0.3, public index | **53.19** (raw 65.23) |
| Requests | 140,178 scoreable, all `ok`, 0 unsupported, 0 errors |
| Request time, one request at a time | median 25.5 · mean 49.1 · p80 31.9 ms |
| Hardware | 1 x NVIDIA RTX PRO 6000 Blackwell (shared with other jobs during the run) |
| Model | [`GyrTechnology/saif-0.8b-lightning`](https://huggingface.co/GyrTechnology/saif-0.8b-lightning/tree/c79f0d24e55685939586a088ea9535a1f3e0523d) (tag `v2.1`, revision `c79f0d24e55685939586a088ea9535a1f3e0523d`) |

## Engine

`decision_index_engine:SaifEngine`, shipped in the model repo with its loader (`saif_v2_core.py`, `singlepass.py`); engine sha256 `94200c22f2930e16e42e2ef976c650439d4111f8013e15bba384af50a2e93672`. In-process, bf16, K=1, batch 1. No prefix caching, no generation, no truncation. Kit `runner.py` sha256 `a1758e8be65cb9a6b44feb1b112e19c25b871451820b2a083540fde964409748`.

## Declared limits

None. Every request in the public suite was answered; nothing was truncated. Text only.

## Results

[runs/saif-0.8b-lightning-v2.1](https://huggingface.co/datasets/GyrTechnology/decision-index-results/tree/00d9c428ce5bb621060eb9c4d300a2ba8dfab38b/runs/saif-0.8b-lightning-v2.1)
