# Crivo-32M

[tardellirs/crivo-32m](https://huggingface.co/tardellirs/crivo-32m) is a 32M-parameter typed-decision encoder,
fine-tuned from `cross-encoder/ettin-reranker-32m-v1` with knowledge distillation from
[Crivo-150M](https://huggingface.co/tardellirs/crivo-150m) (submitted in #127). Each question is read as one sequence
(`[CLS] instructions [SEP] state [SEP] [OPT] option …`) and every option is scored from the hidden state at its `[OPT]`
marker; a softmax over the options gives the distribution. It does not generate text.

| | Crivo-32M |
|---|---:|
| Decision Index 0.3, public index | **25.06** |
| Area skill (0.3) | Knowledge & Reasoning 2.0 · Language Understanding 25.6 · Retrieval & Classification 46.7 · Tools & Automation 36.7 · Arts & Human Taste 18.5 |
| Requests | 140,178 `ok`, 0 `unsupported`, 0 errors (`scores.json`: `"complete": true`) |
| Request time, 1× NVIDIA A100 40 GB, one request at a time | median 10.3 ms · mean 12.6 ms · p95 16.7 ms |
| Results | [runs/crivo-32m](https://huggingface.co/datasets/tardellirs/decision-index-results/tree/eaf516f3d4eea9935cf8f3ae5d6688499498edd4/runs/crivo-32m) (compact: no payloads, no suite text) |
| Weights | `model.safetensors` sha256 `69c1667b4116a451b56262cf643828ecebc41c444b5086fd4c0fee654d5b091f` |

The run used kit `9eb2dbe` (edition 0.3), in one pass, with the engine shipped in the model repository:

```sh
hf download tardellirs/crivo-32m --revision bbf3f1924523f530b3a9e621f2f20f314ea462aa --local-dir crivo-32m
cd crivo-32m
python -m decision_index pipeline --engine decision_index_engine:DecisionEngine --edition 0.3 --out runs/crivo-32m
```

The engine's defaults are the settings of this run: one option order (`tta=1`), fp32 scoring head, SDPA attention, bf16
autocast, `batch_size=32`, `token_budget=65536`, `max_length=32768`, **no truncation**. The encoder was trained up to
8,192 tokens; longer questions (ToolRet tool documents of ~23k tokens) are read whole with rotary positions up to 32,768
instead of being refused. A question above `max_length` would raise `Unsupported`; none exists in the 0.3 suite.
Latency was measured on an A100 40 GB, not the board's RTX PRO 6000.

**Training data.** The same open data as Crivo-150M (Bekko System One dataset v0, train/dev splits of public datasets
behind the index benchmarks rendered with the kit's builders, tasksource procedural typed decisions, ToolACE/glaive tool
selection, CLadder variants not used by the suite, SEntFiN, synthetic smart-home cases), plus format-invariance
augmentations, 328 new task families from tasksource-jev-typed-decisions, and original-vs-perturbed public-domain poems
and prose. Distillation targets come from Crivo-150M; no Jev outputs were used. Nothing comes from evaluation splits.

**Contamination checks.** Leaf-level filter against the rebuilt pools of all 42 benchmarks, builder-level checks of the
public-source train splits, and an exact check of every training question against all 563,686 questions of the final 0.3
suite (0 matches). CLadder training questions sharing ≥90% of their 8-grams with a suite question were also removed.

License: CC BY-NC 4.0.
