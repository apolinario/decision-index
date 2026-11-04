# Solon 1.0 (27B)

[Solon 1.0](https://huggingface.co/Sciforium/Solon-1.0) is [Sciforium](https://www.sciforium.ai/)'s highly optimized decision model, based on Qwen 3.8 27B and trained with supervised fine-tuning. It returns one probability for every option of a question about a state. It does not generate free text.

| | Solon 1.0 (27B) |
|---|---:|
| Decision Index 0.3, public index | **64.76** (raw 73.24) |
| Requests | 140,620: 140,620 `ok` (140,178 scoreable), 0 unsupported, 0 errors |
| Request time, one request at a time per server (scoring run) | median 65 · mean 116 · p80 94 ms |
| Hardware | AMD MI355X (288 GB) |
| Model | [`Sciforium/Solon-1.0`](https://huggingface.co/Sciforium/Solon-1.0) |
| Results | [solon-1.0](https://huggingface.co/datasets/sciforium/solon-1.0-decision-index-run/tree/main/solon-1.0) |

## Engine

The kit's `http` engine against stock SGLang, `POST /v1/systemone`. There is no custom inference code.
- Each request sends the state and questions unmodified, as one `/v1/systemone` call.
- The server is SGLang v0.5.21 ROCm (`lmsysorg/sglang-rocm:v0.5.21-rocm724-mi35x-20261013`), with BF16 weights, triton attention, `--trust-remote-code`, and the tokenizer's own chat template (no `--chat-template` override).
- Kit `runner.py` sha256 `f4c9f36c4c557f537f1d8184db7bf00c5a84d9f4f0aa67f8c06198cead25818d`.

## Declared limits

None. Every request in the public suite was answered, the largest with 64 questions, 151 options and 263,828 input tokens. Nothing was truncated.

## Reproduce

`serve.sh` in the results folder starts one SGLang server on a single NVIDIA RTX PRO 6000 (96 GB). It uses the same server flags as the submitted run, plus two caps for the 96 GB card: `--max-running-requests 16` and `--max-mamba-cache-size 16`, because the hybrid GDN model keeps mamba state for each running request. `--context-length` stays at the model default so no request is refused.

The [results HF repo](https://huggingface.co/datasets/sciforium/solon-1.0-decision-index-run) README has more details.

```bash
./serve.sh
python -m decision_index pipeline --engine http \
  --option base_url=http://127.0.0.1:30000 --option model=Sciforium/Solon-1.0 \
  --suite-dir <path-to>/suite-0.3 --out solon-1.0
```

Please refer to the model card: [Sciforium/Solon-1.0](https://huggingface.co/Sciforium/Solon-1.0).
