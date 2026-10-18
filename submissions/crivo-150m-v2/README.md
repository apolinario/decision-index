# Crivo-150M v2

**This is an update of an existing board entry, not a new model.** It replaces the board entry Crivo-150M (`crivo-150m`,
#127; v1 at revision `7a6c745`, public 36.39, Full 22.21) with a newer version of the same model, published as a new
revision of the same repository: [tardellirs/crivo-150m](https://huggingface.co/tardellirs/crivo-150m/tree/a5e21d8067c5c37ba5f88cd498306ae5abfd625d)
at `a5e21d8067c5c37ba5f88cd498306ae5abfd625d`.

Crivo-150M v2 is a 149.6M-parameter typed-decision encoder, fine-tuned from `cross-encoder/ettin-reranker-150m-v1`
with knowledge distillation from Crivo-400M (#167), on the same training data. Architecture, size, input format and
interface are the same as v1; the weights, the training data and the teacher are new. Each question is read as one
sequence (`[CLS] instructions [SEP] state [SEP] [OPT] option …`) and every option is scored from the hidden state at its
`[OPT]` marker; a softmax over the options gives the distribution. It does not generate text.

| | Crivo-150M v2 |
|---|---:|
| Decision Index 0.3, public index | **34.38** (raw 51.32) |
| Area skill (0.3) | Knowledge & Reasoning 8.3 · Language Understanding 34.5 · Retrieval & Classification 48.2 · Tools & Automation 62.8 · Arts & Human Taste 21.8 |
| Requests | 140,178 `ok`, 0 `unsupported`, 0 errors (`scores.json`: `"complete": true`) |
| Request time, 1× NVIDIA A100 40 GB, one request at a time | median 18.0 ms · mean 26.1 ms · p95 44.4 ms |
| Results | [runs/crivo-150m-v2](https://huggingface.co/datasets/tardellirs/decision-index-results/tree/7ba8f95cf5900e430ef09951b7cbd1eece741903/runs/crivo-150m-v2) (compact: no payloads, no suite text) |
| Weights | `model.safetensors` sha256 `9022e370bb7ddf91297a64c0b538540acdd5c09dee5848ea854bb177577daab8` |
| Parameters | 149,610,241 (unchanged from v1) |
| Previous version (v1) | revision `7a6c7453c06be576e1b48aeb70429318402791d7`, `model.safetensors` sha256 `011a550c8e4d0cb275240919a875f5e489e5c85dde5c1002608f5a804883387b`; public 36.39 (raw 52.45) |

The run used kit `9eb2dbe` (edition 0.3), in one pass, with the engine shipped in the model repository:

```sh
hf download tardellirs/crivo-150m --revision a5e21d8067c5c37ba5f88cd498306ae5abfd625d --local-dir crivo-150m-v2
cd crivo-150m-v2
python -m decision_index pipeline --engine decision_index_engine:DecisionEngine --edition 0.3 --out runs/crivo-150m-v2
```

The engine's defaults are the settings of this run: one option order (`tta=1`), fp32 scoring head, SDPA attention, bf16
autocast, `batch_size=32`, `token_budget=65536`, `max_length=32768`, **no truncation**. The encoder was trained up to
8,192 tokens; longer questions (ToolRet tool documents of ~23k tokens) are read whole with rotary positions up to 32,768
instead of being refused. A question above `max_length` would raise `Unsupported`; none exists in the 0.3 suite.
The engine is the same file as Crivo-400M's (it replaces v1's engine; the main difference is the fp32 scoring head).
Latency was measured on an A100 40 GB, not the board's RTX PRO 6000; the suite ran after the other job on that GPU had
finished. The machine's Python was 3.10, so `hashlib.file_digest` in `suite/io.py` was replaced by the equivalent
chunked sha256 for that run; nothing else in the kit was changed. The uploaded `results.jsonl.gz` drops `payload` and
`raw_output` (as `--compact` does); rescored with the kit it gives the same 34.38. On the package, two runs of 500
questions with `batch_size` 32 and 4 gave the same answer on every question (max probability difference 0).

**What changed from v1.** v1 trained on ~210k rows of the benchmarks' own train splits in the suite's layout and had no
teacher. v2 uses the training files of Crivo-400M, where each benchmark source is capped at 5,000 rows (about half
rewritten into other layouts) and most of the data is other task families, and is distilled from Crivo-400M. Its public
index is lower (36.39 → 34.38); on our own held-out selection panels it is higher on both the new-domain families and
the development set of the board's skills. The model card has the per-benchmark comparison.

**Training data.** Exactly the training files of Crivo-400M: Bekko System One dataset v0, code-generated decision
families with exact labels, format-invariance augmentations, tasksource-jev-typed-decisions and procedural typed
decisions, sibling sources of the suite's skills (argument quality, SEntFiN, RAGBench, Hermes function calling, xLAM
irrelevance, bitext intents, CUAD), train/dev splits of public datasets behind the index benchmarks (original layout and
rewritten), open teacher labels from Mapika/decider (open Qwen models), CFPB complaints, CommitPackFT, ToolACE/glaive
tool selection, CLadder variants not used by the suite, synthetic smart-home cases, and synthetic "which API next"
dialogues built from Hermes function calling. The API-Bank train split was not used, because it repeats dialogues and
catalogs of the test pool. No Jev outputs were used, and nothing comes from evaluation splits.

**Distillation.** In stages 1 and 2 the loss mixes the gold label and Crivo-400M's logits on the same training rows
(α = 0.5); sources where the teacher agrees with the gold label on fewer than 60% of their rows, plus smart-home control,
CLadder and RAGTruth, train on gold only, and stage 3 (long context) is gold only. The released weights are the end of
stage 3, chosen on our own held-out panels, never on the board; the complete public suite was run only on the final
package.

**Contamination checks.** Every training example was checked against the 0.3 suite rows (same normalized state and
question, same distinctive state, or a shared passage of ≥120 characters were dropped), and a final audit of the exact
training files compared every training question against all 563,686 questions of the 0.3 suite: 0 matches. The teacher
only labeled these training rows.

License: CC BY-NC 4.0 (unchanged).
