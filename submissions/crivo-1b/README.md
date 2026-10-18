# Crivo-1B

[tardellirs/crivo-1b](https://huggingface.co/tardellirs/crivo-1b) is a 1.03B-parameter typed-decision encoder (28
layers, hidden 1792), fine-tuned from `cross-encoder/ettin-reranker-1b-v1` on the training data of Crivo-400M (#167),
with the same three-stage recipe and no teacher. It is the largest of the Crivo family. Each question is read as one
sequence (`[CLS] instructions [SEP] state [SEP] [OPT] option …`) and every option is scored from the hidden state at its
`[OPT]` marker; a softmax over the options gives the distribution. It does not generate text.

| | Crivo-1B |
|---|---:|
| Decision Index 0.3, public index | **47.77** (raw 61.29) |
| Area skill (0.3) | Knowledge & Reasoning 18.5 · Language Understanding 56.3 · Retrieval & Classification 55.8 · Tools & Automation 78.7 · Arts & Human Taste 28.9 |
| Requests | 140,178 `ok`, 0 `unsupported`, 0 errors (`scores.json`: `"complete": true`) |
| Request time, 1× NVIDIA H100 PCIe 80 GB, one request at a time | median 13.2 ms · mean 36.4 ms · p95 94.1 ms |
| Results | [runs/crivo-1b](https://huggingface.co/datasets/tardellirs/decision-index-results/tree/7ba8f95cf5900e430ef09951b7cbd1eece741903/runs/crivo-1b) (compact: no payloads, no suite text) |
| Weights | `model.safetensors` sha256 `2ec762c12224a15a244f02f3e80b2bf2dba29eb448405f49ddbcb1ff96dfd854` |
| Parameters | 1,031,276,289 |

The run used kit `9eb2dbe` (edition 0.3), in one pass. To reproduce it with the engine shipped in the model repository:

```sh
hf download tardellirs/crivo-1b --revision 5dab4fb0ee59bbd08d0126027aff14ddf90466ee --local-dir crivo-1b
cd crivo-1b
python -m decision_index pipeline --engine decision_index_engine:DecisionEngine --edition 0.3 --out runs/crivo-1b
```

The engine's defaults are the settings of this run: one option order (`tta=1`), fp32 scoring head, SDPA attention, bf16
autocast, `batch_size=32`, `token_budget=65536`, `max_length=32768`, **no truncation**. The encoder was trained up to
8,192 tokens; longer questions (ToolRet tool documents of ~23k tokens) are read whole with rotary positions up to 32,768
instead of being refused. A question above `max_length` would raise `Unsupported`; none exists in the 0.3 suite.
The engine is the same file as Crivo-400M's.

**How this run was invoked.** Unlike its siblings, the suite ran right after training, on the training box, from the
final checkpoint directory and with the engine under its development module name, so `environment.json` and
`scores.json` show `engine2:DecisionEngine2` with `path=../runs/c1bs3` and `head_fp32=true`. The released package was
then built from that directory: `model.safetensors` is a byte-for-byte copy of the checkpoint, and
`decision_index_engine.py` is the same code as `engine2.py` plus the alias `DecisionEngine = DecisionEngine2`, with
`path` defaulting to the repository folder and `head_fp32=True` already the default. The results are therefore those of
the published package. On the final checkpoint, two runs of 500 suite questions with `batch_size` 32 and 4 gave the same
answer on every question (max probability difference 0).

Latency was measured on an H100 PCIe 80 GB, not the board's RTX PRO 6000, and is not comparable with the A100 numbers
of the smaller Crivo entries. The run script replaces `hashlib.file_digest` in `suite/io.py` with the equivalent
chunked sha256 when Python is older than 3.11 (the GPU image ships Python 3.10); nothing else in the kit was changed.
The uploaded `results.jsonl.gz` drops `payload` and `raw_output` (as `--compact` does); rescored with the kit it gives
the same 47.77.

**Training data.** Exactly the training files of Crivo-400M: Bekko System One dataset v0, code-generated decision
families with exact labels, format-invariance augmentations, tasksource-jev-typed-decisions and procedural typed
decisions, sibling sources of the suite's skills (argument quality, SEntFiN, RAGBench, Hermes function calling, xLAM
irrelevance, bitext intents, CUAD), train/dev splits of public datasets behind the index benchmarks (original layout and
rewritten), open teacher labels from Mapika/decider (open Qwen models), CFPB complaints, CommitPackFT, ToolACE/glaive
tool selection, CLadder variants not used by the suite, synthetic smart-home cases, and synthetic "which API next"
dialogues built from Hermes function calling. The API-Bank train split was not used, because it repeats dialogues and
catalogs of the test pool. No Jev outputs were used, and nothing comes from evaluation splits.

**Recipe.** Three stages, one epoch each, gold labels only (no larger Crivo exists to serve as teacher): stage 1 with
weight EMA 0.999, stage 2 from the stage-1 EMA, stage 3 at 8,192 tokens. The end of stage 2, the end of stage 3 and
their average were compared on our own held-out panels, never on the board, with protection gates relative to
Crivo-150M v1; the average was marginally higher on the panels but failed the gate on our RAGTruth development set, so
the released weights are the end of stage 3. The complete public suite was run only on that checkpoint.

**Contamination checks.** Every training example was checked against the 0.3 suite rows (same normalized state and
question, same distinctive state, or a shared passage of ≥120 characters were dropped), and a final audit of the exact
training files compared every training question against all 563,686 questions of the 0.3 suite: 0 matches.

License: CC BY-NC 4.0.
