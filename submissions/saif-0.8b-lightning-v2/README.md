# Saif 0.8B Lightning v2 (0.8B)

[Saif 0.8B Lightning](https://huggingface.co/GyrTechnology/saif-0.8b-lightning/tree/1ba11d5dc6233a2bf35f7172a89137c5b63bb78f) is GyrTechnology's small decision model, a full fine-tune of Qwen3.5-0.8B. It returns one probability for every option of every question in a single forward pass. It does not generate text. v2 replaces v0.3 (public index 47.75); it is a clean retrain on public training splits only.

| | Saif 0.8B Lightning v2 |
|---|---:|
| Decision Index 0.3, public index | **53.65** (raw 65.35) |
| Requests | 140,178 `ok` (140,178 scoreable), 0 unsupported, 0 errors |
| Request time, one request at a time (scoring run) | median 26.0 · mean 58.5 · p80 32.4 ms |
| Hardware | 1 x NVIDIA GeForce RTX 5070 Ti (16 GB) |
| Model | [`GyrTechnology/saif-0.8b-lightning`](https://huggingface.co/GyrTechnology/saif-0.8b-lightning/tree/1ba11d5dc6233a2bf35f7172a89137c5b63bb78f) (tag `v2`, revision `1ba11d5dc6233a2bf35f7172a89137c5b63bb78f`) |
| Results | [runs/saif-0.8b-lightning-v2](https://huggingface.co/datasets/GyrTechnology/decision-index-results/tree/a750a06442e54aef0e509f9894004c9a344d1a48/runs/saif-0.8b-lightning-v2) |

## Engine

`decision_index_engine:SaifEngine`, shipped in the model repo with its loader (`saif_v2_core.py`, `singlepass.py`); engine sha256 `9a0af82e4ae1a3dd55f37dc7a04e4e5702df0df29f2f99f036098513ab88fa51`. In-process, bf16, K=1, batch 1, one process. The state is read once and each question in its own segment over the shared state; no prefix caching, no generation, no truncation. Kit `runner.py` sha256 `f4c9f36c4c557f537f1d8184db7bf00c5a84d9f4f0aa67f8c06198cead25818d`.

## Declared limits

None. Every request in the public suite was answered; nothing was truncated. Text only.

## Reproduce

```bash
git clone --branch v2 https://huggingface.co/GyrTechnology/saif-0.8b-lightning
PYTHONPATH=saif-0.8b-lightning:<path-to>/decision-index python -m decision_index pipeline --edition 0.3 \
  --suite-dir <path-to>/suite-0.3 --engine decision_index_engine:SaifEngine \
  --model saif-0.8b-lightning --out saif-0.8b-lightning --compact
```

Requirements: Python 3.12, torch 2.8, transformers with Qwen3.5 support, safetensors, accelerate. Please refer to the model card: [GyrTechnology/saif-0.8b-lightning](https://huggingface.co/GyrTechnology/saif-0.8b-lightning).
