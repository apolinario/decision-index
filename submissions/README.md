# Submissions

| Model | Results | Engine / source | Hardware |
|---|---|---|---|
| Reflex 27B (Qwen3.8-27B-FP8 @ `017b9c7`, two orderings, question preview + reading rules) | [Complete Decision Index 0.3 run](https://huggingface.co/datasets/Kshetrajna/reflex-27b-pmscr-decision-index-03/blob/29ba3778540115c07d840b2b7aedef0c352d9f7b/runs/reflex-27b-pmscr-di03/scores.json) | [Reflex `di03-main-pmscr2`](https://github.com/kshetrajna12/reflex/tree/di03-main-pmscr2) (`5e48585`), prompt texts `configs/decision-index/pmscr.json`; SGLang `lmsysorg/sglang@sha256:b27fce60…`; kit `reflex_http`, `permutations: 2`, `timeout 600`, 32 workers | H100 80GB (0.2.1 run: 4 disjoint shards, one per GPU; 0.3 GSM8K resume: 1 GPU) |

Reflex 27B scored 57.34 on the 0.3 public index, with all 140,178 scoreable requests answered (0 errors, 0 unsupported). The context limit is 65,536 tokens and choice width is limited by the tokenizer's one-token labels; nothing was truncated or refused. Launch commands: `docs/decision-index-0.3.md` at the tag. The [dataset](https://huggingface.co/datasets/Kshetrajna/reflex-27b-pmscr-decision-index-03/tree/29ba3778540115c07d840b2b7aedef0c352d9f7b) has predictions, the exact serving source, runtime and H100 latency (median 142 / mean 194 / p80 207 ms, single client).
