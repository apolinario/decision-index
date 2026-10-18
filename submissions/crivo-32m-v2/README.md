# Crivo-32M v2

**This is an update of an existing board entry, not a new model.** It replaces the board entry Crivo-32M (`crivo-32m`,
#131; v1 at revision `bbf3f19`, public 25.06, Full 11.87) with a newer version of the same model, published as a new
revision of the same repository: [tardellirs/crivo-32m](https://huggingface.co/tardellirs/crivo-32m/tree/93970f0562b85138e2c44b65ed0bc46a6d082f14)
at `93970f0562b85138e2c44b65ed0bc46a6d082f14`.

Crivo-32M v2 is a 32.0M-parameter typed-decision encoder, fine-tuned from `cross-encoder/ettin-reranker-32m-v1` with
knowledge distillation from Crivo-400M (#167), on the same training data (v1 was distilled from Crivo-150M v1).
Architecture, size, input format and interface are the same as v1; the weights, the training data and the teacher are
new. Each question is read as one sequence (`[CLS] instructions [SEP] state [SEP] [OPT] option …`) and every option is
scored from the hidden state at its `[OPT]` marker; a softmax over the options gives the distribution. It does not
generate text.

| | Crivo-32M v2 |
|---|---:|
| Decision Index 0.3, public index | **27.34** (raw 45.75) |
| Area skill (0.3) | Knowledge & Reasoning 2.7 · Language Understanding 25.0 · Retrieval & Classification 45.6 · Tools & Automation 50.0 · Arts & Human Taste 19.0 |
| Requests | 140,178 `ok`, 0 `unsupported`, 0 errors (`scores.json`: `"complete": true`) |
| Request time, 1× NVIDIA A100 40 GB, one request at a time | median 26.9 ms · mean 30.5 ms · p95 41.3 ms, **with the GPU shared with another training job** (see below) |
| Results | [runs/crivo-32m-v2](https://huggingface.co/datasets/tardellirs/decision-index-results/tree/7ba8f95cf5900e430ef09951b7cbd1eece741903/runs/crivo-32m-v2) (compact: no payloads, no suite text) |
| Weights | `model.safetensors` sha256 `60f361d2c02a19177be2474ac6dd62b62a44f49982f7949eb6d88d9720b71b39` |
| Parameters | 32,033,665 (unchanged from v1) |
| Previous version (v1) | revision `bbf3f1924523f530b3a9e621f2f20f314ea462aa`, `model.safetensors` sha256 `69c1667b4116a451b56262cf643828ecebc41c444b5086fd4c0fee654d5b091f`; public 25.06 (raw 43.20) |

The run used kit `9eb2dbe` (edition 0.3), in one pass, with the engine shipped in the model repository:

```sh
hf download tardellirs/crivo-32m --revision 93970f0562b85138e2c44b65ed0bc46a6d082f14 --local-dir crivo-32m-v2
cd crivo-32m-v2
python -m decision_index pipeline --engine decision_index_engine:DecisionEngine --edition 0.3 --out runs/crivo-32m-v2
```

The engine's defaults are the settings of this run: one option order (`tta=1`), fp32 scoring head, SDPA attention, bf16
autocast, `batch_size=32`, `token_budget=65536`, `max_length=32768`, **no truncation**. The released weights were
trained on sequences up to 2,048 tokens; longer questions (for example ToolRet tool documents of ~23k tokens) are read
whole with rotary positions up to 32,768 instead of being refused. A question above `max_length` would raise
`Unsupported`; none exists in the 0.3 suite. The engine is the same file as Crivo-400M's. The machine's Python was 3.10,
so `hashlib.file_digest` in `suite/io.py` was replaced by the equivalent chunked sha256 for that run; nothing else in
the kit was changed. The uploaded `results.jsonl.gz` drops `payload` and `raw_output` (as `--compact` does); rescored
with the kit it gives the same 27.34. On the package, two runs of 500 questions with `batch_size` 32 and 4 gave the
same answer on every question (max probability difference 0).

**Latency note.** This suite ran on an A100 40 GB that was also training Crivo-150M v2 at the time, so the request
times above are inflated and should not be compared with other entries. The network and the engine's batching are the
same as v1's (median 10.3 ms on an unshared A100 40 GB; 7.3 ms on the board's RTX PRO 6000).

**What changed from v1.** v1 trained on ~209k rows of the benchmarks' own train splits in the suite's layout, with
Crivo-150M v1 as teacher. v2 uses the training files of Crivo-400M, where each benchmark source is capped at 5,000 rows
(about half rewritten into other layouts) and most of the data is other task families, and is distilled from
Crivo-400M. Its public index is higher (25.06 → 27.34, mostly tools and API selection), and on our own held-out
selection panels it is higher on both the new-domain families and the development set of the board's skills. The model
card has the per-benchmark comparison.

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
CLadder and RAGTruth, train on gold only. The released weights are the end of stage 2 (a long-context stage 3 was
trained but scored lower), chosen on our own held-out panels, never on the board; the complete public suite was run
only on the final package.

**Contamination checks.** Every training example was checked against the 0.3 suite rows (same normalized state and
question, same distinctive state, or a shared passage of ≥120 characters were dropped), and a final audit of the exact
training files compared every training question against all 563,686 questions of the 0.3 suite: 0 matches. The teacher
only labeled these training rows.

License: CC BY-NC 4.0 (unchanged).
