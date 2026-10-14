# Ventus-2B (Ethan33X)

[Ethan33X/Ventus-2B](https://huggingface.co/Ethan33X/Ventus-2B) is a 2.48B-parameter open-weights decision foundation model fine-tuned on top of Qwen3.5-2B. It evaluates in a single forward pass via direct causal option-letter logits using standard Hugging Face AutoModelForCausalLM. Released under the Apache-2.0 License. It does not generate free-form text.

| | Ventus-2B |
|---|---:|
| Decision Index 0.3, public index | **13.85** (raw 33.85) |
| Area skill: knowledge · language · retrieval · tools · arts | 9.54 · 14.30 · 5.65 · 31.05 · 8.78 |
| Requests | 140,178 scoreable; 140,174 ok; 0 unsupported; 4 errors |
| Median request time (4× RTX 4080 32 GB, this run) | 41.6 ms (mean 148.3 ms, p95 387.7 ms) |
| Model Repository | [Ethan33X/Ventus-2B](https://huggingface.co/Ethan33X/Ventus-2B) |
| Inference Code | [XieShaocong33Ethan/Ventus-2B](https://github.com/XieShaocong33Ethan/Ventus-2B) |
| Results Dataset | [runs/ventus-2b](https://huggingface.co/datasets/Ethan33X/Ventus-2B-decision-index/tree/main/runs/ventus-2b) |

The results are compact, with zero suite text leaked.

## How it was run

```sh
pip install -e ".[transformers,rebuild]"
python -m decision_index suite verify --dir suite-0.3 --edition 0.3
python run_ventus_shards.py --model Ethan33X/Ventus-2B --gpus 0,1,2,3 --edition 0.3 --suite-dir suite-0.3 --out runs/ventus-2b
```

- Kit `9eb2dbe`; the suite rebuilt with `suite rebuild --edition 0.3` and checked by `suite verify` (100% hash match).
- 4× NVIDIA GeForce RTX 4080 (32 GB), bf16, single forward pass evaluating direct option-letter logits over native chat templates with PyTorch SDPA.
- Run completed with `"complete": true` across all 140,178 scoreable requests.

## Declared limits

- Context limit is 32,768 tokens and nothing is truncated. Inputs exceeding context capacity are answered as unsupported. No public request reached this limit.
