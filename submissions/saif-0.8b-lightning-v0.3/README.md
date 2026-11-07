# Saif 0.8B Lightning v0.3 (0.8B)

[Saif 0.8B Lightning](https://huggingface.co/GyrTechnology/saif-0.8b-lightning/tree/0cc7d0954a7aa48e0ba7302c4839185eb5aa164d) is GyrTechnology's small decision model, a full fine-tune of Qwen3.5-0.8B. It returns one probability for every option of every question in a single forward pass. It does not generate text. v0.3 replaces v0.1 (public index 42.79).

| | Saif 0.8B Lightning v0.3 |
|---|---:|
| Decision Index 0.3, public index | **47.75** (raw 60.73) |
| Requests | 140,178 `ok` (140,178 scoreable), 0 unsupported, 0 errors |
| Request time, one request at a time (scoring run) | median 18.1 · mean 54.5 · p80 30.1 ms |
| Hardware | 1 x NVIDIA GeForce RTX 5070 Ti (16 GB) |
| Model | [`GyrTechnology/saif-0.8b-lightning`](https://huggingface.co/GyrTechnology/saif-0.8b-lightning/tree/0cc7d0954a7aa48e0ba7302c4839185eb5aa164d) (tag `v0.3`, revision `0cc7d0954a7aa48e0ba7302c4839185eb5aa164d`) |
| Results | [runs/saif-0.8b-lightning-v0.3](https://huggingface.co/datasets/GyrTechnology/decision-index-results/tree/262df3ad031f8b8004888c50cfdd47011ff7d45d/runs/saif-0.8b-lightning-v0.3) |

## Engine

`decision_index_engine:SaifEngine`, shipped in the model repo with its loader (`saif_v2_core.py`, `singlepass.py`); engine sha256 `7a18d7efe6a337df3fcafb7249d43fa031b8a76860df1625e211d930ff1f736e`. In-process, bf16, K=1, batch 1, one process. The state is read once and each question in its own segment over the shared state; no prefix caching, no generation, no truncation. Kit `runner.py` sha256 `f4c9f36c4c557f537f1d8184db7bf00c5a84d9f4f0aa67f8c06198cead25818d`.

## Declared limits

None. Every request in the public suite was answered; nothing was truncated. Text only.

## Reproduce

```bash
git clone --branch v0.3 https://huggingface.co/GyrTechnology/saif-0.8b-lightning
PYTHONPATH=saif-0.8b-lightning:<path-to>/decision-index python -m decision_index pipeline --edition 0.3 \
  --suite-dir <path-to>/suite-0.3 --engine decision_index_engine:SaifEngine \
  --model saif-0.8b-lightning --out saif-0.8b-lightning --compact
```

Requirements: Python 3.12, torch 2.8, transformers with Qwen3.5 support, safetensors, accelerate. Please refer to the model card: [GyrTechnology/saif-0.8b-lightning](https://huggingface.co/GyrTechnology/saif-0.8b-lightning).
