# Saif 4B v2.1 (4B)

[Saif 4B](https://huggingface.co/GyrTechnology/saif-4b/tree/8319c9c33e88850b2e403378d2d0105957b38a37) is GyrTechnology's decision model, a full fine-tune of Qwen3.5-4B. It returns one probability for
every option of every question in a single forward pass and does not generate text. v2.1 replaces v2.0 (public index
57.71).

| | Saif 4B v2.1 |
|---|---:|
| Decision Index 0.3, public index | **62.94** (raw 72.21) |
| Requests | 140,178 `ok` (140,178 scoreable), 0 unsupported, 0 errors |
| Request time, one request at a time (scoring run) | median 32.0 · mean 107.1 · p80 63.7 ms |
| Hardware | 1 x NVIDIA RTX PRO 6000 Blackwell Max-Q (96 GB) |
| Model | [`GyrTechnology/saif-4b`](https://huggingface.co/GyrTechnology/saif-4b/tree/8319c9c33e88850b2e403378d2d0105957b38a37) (tag `v2.1`, revision `8319c9c33e88850b2e403378d2d0105957b38a37`) |
| Results | [runs/saif-4b-v2.1](https://huggingface.co/datasets/GyrTechnology/decision-index-results/tree/7fc63b1f13688f44e9cd3171ad911b4e660125ce/runs/saif-4b-v2.1) |

## Engine

`decision_index_engine:SaifEngine`, shipped in the model repo with its loader (`saif_v2_core.py`, `singlepass.py`);
engine sha256 `9a0af82e4ae1a3dd55f37dc7a04e4e5702df0df29f2f99f036098513ab88fa51`. In-process, bf16, K=1, batch 1, one process. The state is read once and each question in
its own segment over the shared state; no prefix caching, no generation, no truncation. Kit `runner.py` sha256
`f4c9f36c4c557f537f1d8184db7bf00c5a84d9f4f0aa67f8c06198cead25818d`.

## Declared limits

None. Every request in the public suite was answered; nothing was truncated. Text only.

## Reproduce

```bash
git clone --branch v2.1 https://huggingface.co/GyrTechnology/saif-4b
PYTHONPATH=saif-4b:<path-to>/decision-index python -m decision_index pipeline --edition 0.3 \
  --suite-dir <path-to>/suite-0.3 --engine decision_index_engine:SaifEngine \
  --model saif-4b --out saif-4b --compact
```

Requirements: Python 3.12, torch 2.8, transformers with Qwen3.5 support, safetensors, accelerate.
