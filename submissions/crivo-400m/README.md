# Crivo-400M

[tardellirs/crivo-400m](https://huggingface.co/tardellirs/crivo-400m) is a 396M-parameter typed-decision encoder,
fine-tuned from `cross-encoder/ettin-reranker-400m-v1`, the larger sibling of Crivo-150M (#127) and Crivo-32M (#131).
Each question is read as one sequence (`[CLS] instructions [SEP] state [SEP] [OPT] option …`) and every option is
scored from the hidden state at its `[OPT]` marker; a softmax over the options gives the distribution. It does not
generate text.

| | Crivo-400M |
|---|---:|
| Decision Index 0.3, public index | **39.79** (raw 55.55) |
| Area skill (0.3) | Knowledge & Reasoning 11.5 · Language Understanding 44.3 · Retrieval & Classification 51.0 · Tools & Automation 69.0 · Arts & Human Taste 25.4 |
| Requests | 140,178 `ok`, 0 `unsupported`, 0 errors (`scores.json`: `"complete": true`) |
| Request time, 1× NVIDIA A100 40 GB, one request at a time | median 23.4 ms · mean 38.1 ms · p95 73.6 ms |
| Results | [runs/crivo-400m](https://huggingface.co/datasets/tardellirs/decision-index-results/tree/6d9fb5a6a10c71069442d50954a553dad280a9c8/runs/crivo-400m) (compact: no payloads, no suite text) |
| Weights | `model.safetensors` sha256 `3e4ada0f685e4dbeb0967bc40c4e6b3b68f83dfe8c9f6458e4f6eabe29b2c316` |

The run used kit `9eb2dbe` (edition 0.3), in one pass, with the engine shipped in the model repository:

```sh
hf download tardellirs/crivo-400m --revision 81f9cab8d7b3cc049b33033952850bb15ce654ae --local-dir crivo-400m
cd crivo-400m
python -m decision_index pipeline --engine decision_index_engine:DecisionEngine --edition 0.3 --out runs/crivo-400m
```

The engine's defaults are the settings of this run: one option order (`tta=1`), fp32 scoring head, SDPA attention, bf16
autocast, `batch_size=32`, `token_budget=65536`, `max_length=32768`, **no truncation**. The encoder was trained up to
8,192 tokens; longer questions (ToolRet tool documents of ~23k tokens) are read whole with rotary positions up to 32,768
instead of being refused. A question above `max_length` would raise `Unsupported`; none exists in the 0.3 suite.
Latency was measured on an A100 40 GB, not the board's RTX PRO 6000. The machine's Python was 3.10, so
`hashlib.file_digest` in `suite/io.py` was replaced by the equivalent chunked sha256 for that run; nothing else in the kit
was changed. The uploaded `results.jsonl.gz` drops `payload` and `raw_output` (as `--compact` does); rescored with the
kit it gives the same 39.79.

**Training data.** Open data only: Bekko System One dataset v0, code-generated decision families with exact labels,
format-invariance augmentations, tasksource-jev-typed-decisions and procedural typed decisions, sibling sources of the
suite's skills (argument quality, SEntFiN, RAGBench, Hermes function calling, xLAM irrelevance, bitext intents, CUAD),
train/dev splits of public datasets behind the index benchmarks (original layout and rewritten), open teacher labels
from Mapika/decider (open Qwen models), CFPB complaints, CommitPackFT, ToolACE/glaive tool selection, CLadder variants not
used by the suite, synthetic smart-home cases, and synthetic "which API next" dialogues built from Hermes function
calling. The API-Bank train split was not used, because it repeats dialogues and catalogs of the test pool. No Jev
outputs were used, and nothing comes from evaluation splits.

**Contamination checks.** Every training example was checked against the 0.3 suite rows (same normalized state and
question, same distinctive state, or a shared passage of ≥120 characters were dropped), and a final audit of the exact
training files compared every training question against all 563,686 questions of the 0.3 suite: 0 matches.
