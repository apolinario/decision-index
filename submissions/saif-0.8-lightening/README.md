# Saif 0.8 Lightening v0.1 (0.8B)

[Saif 0.8 Lightening](https://huggingface.co/GyrTechnology/saif-0.8-lightening/tree/e56bc26ee29be5e1ce42aba0f1517cdaeb721c7a) is GyrTechnology's small decision model, a full fine-tune of Qwen3.5-0.8B. It returns one probability for every option of every question in a single forward pass. It does not generate text.

| | Saif 0.8 Lightening v0.1 |
|---|---:|
| Decision Index 0.3, public index | **42.79** (raw 56.65) |
| Requests | 140,620: 140,620 `ok` (140,178 scoreable), 0 unsupported, 0 errors |
| Request time, one request at a time (scoring run) | median 18.2 · mean 54.6 · p80 30.5 ms |
| Hardware | 1 x NVIDIA GeForce RTX 5070 Ti (16 GB) |
| Model | [`GyrTechnology/saif-0.8-lightening`](https://huggingface.co/GyrTechnology/saif-0.8-lightening/tree/e56bc26ee29be5e1ce42aba0f1517cdaeb721c7a) (tag `v0.1`, revision `e56bc26ee29be5e1ce42aba0f1517cdaeb721c7a`) |
| Results | [runs/saif-0.8-lightening-v0.1](https://huggingface.co/datasets/GyrTechnology/decision-index-results/tree/c06576dd5fd32f2a25f1b3981e9eb38d30634ac7/runs/saif-0.8-lightening-v0.1) |

## Engine

`decision_index_engine:SaifEngine`, shipped in the model repo with its loader (`saif_v2_core.py`, `singlepass.py`); engine sha256 `7a18d7efe6a337df3fcafb7249d43fa031b8a76860df1625e211d930ff1f736e`. In-process, bf16, K=1, batch 1, one process. The state is read once and each question in its own segment over the shared state; no prefix caching, no generation, no truncation. Kit `runner.py` sha256 `f4c9f36c4c557f537f1d8184db7bf00c5a84d9f4f0aa67f8c06198cead25818d`.

## Declared limits

None. Every request in the public suite was answered; nothing was truncated. Text only.

## Reproduce

```bash
git clone https://huggingface.co/GyrTechnology/saif-0.8-lightening
PYTHONPATH=saif-0.8-lightening:<path-to>/decision-index python -m decision_index pipeline --edition 0.3 \
  --suite-dir <path-to>/suite-0.3 --engine decision_index_engine:SaifEngine \
  --model saif-0.8-lightening --out saif-0.8-lightening --compact
```

Requirements: Python 3.12, torch 2.8, transformers with Qwen3.5 support, safetensors, accelerate. Please refer to the model card: [GyrTechnology/saif-0.8-lightening](https://huggingface.co/GyrTechnology/saif-0.8-lightening).
