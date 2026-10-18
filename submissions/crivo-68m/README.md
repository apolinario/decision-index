# Crivo-68M

[tardellirs/crivo-68m](https://huggingface.co/tardellirs/crivo-68m) is a 68M-parameter typed-decision encoder,
fine-tuned from `cross-encoder/ettin-reranker-68m-v1` with knowledge distillation from Crivo-400M (#167), on the same
training data. It sits between Crivo-32M (#131) and Crivo-150M (#127) in the Crivo family. Each question is read as one
sequence (`[CLS] instructions [SEP] state [SEP] [OPT] option …`) and every option is scored from the hidden state at its
`[OPT]` marker; a softmax over the options gives the distribution. It does not generate text.

| | Crivo-68M |
|---|---:|
| Decision Index 0.3, public index | **32.82** (raw 50.15) |
| Area skill (0.3) | Knowledge & Reasoning 5.0 · Language Understanding 30.8 · Retrieval & Classification 48.8 · Tools & Automation 64.0 · Arts & Human Taste 20.9 |
| Requests | 140,178 `ok`, 0 `unsupported`, 0 errors (`scores.json`: `"complete": true`) |
| Request time, 1× NVIDIA A100 40 GB, one request at a time | median 16.4 ms · mean 21.2 ms · p95 28.6 ms |
| Results | [runs/crivo-68m](https://huggingface.co/datasets/tardellirs/decision-index-results/tree/7ba8f95cf5900e430ef09951b7cbd1eece741903/runs/crivo-68m) (compact: no payloads, no suite text) |
| Weights | `model.safetensors` sha256 `b9b8dcf0ddaf89c4d4a55bf47afeccf78596743a256bfd2e42a0f6961cc1b8e4` |

The run used kit `9eb2dbe` (edition 0.3), in one pass, with the engine shipped in the model repository:

```sh
hf download tardellirs/crivo-68m --revision 4519d10aa61562f1c74461ff84a45dcda0944e02 --local-dir crivo-68m
cd crivo-68m
python -m decision_index pipeline --engine decision_index_engine:DecisionEngine --edition 0.3 --out runs/crivo-68m
```

The engine's defaults are the settings of this run: one option order (`tta=1`), fp32 scoring head, SDPA attention, bf16
autocast, `batch_size=32`, `token_budget=65536`, `max_length=32768`, **no truncation**. The encoder was trained up to
8,192 tokens; longer questions (ToolRet tool documents of ~23k tokens) are read whole with rotary positions up to 32,768
instead of being refused. A question above `max_length` would raise `Unsupported`; none exists in the 0.3 suite.
The engine is the same file as Crivo-400M's. Latency was measured on an A100 40 GB, not the board's RTX PRO 6000. The
machine's Python was 3.10, so `hashlib.file_digest` in `suite/io.py` was replaced by the equivalent chunked sha256 for
that run; nothing else in the kit was changed. The uploaded `results.jsonl.gz` drops `payload` and `raw_output` (as
`--compact` does); rescored with the kit it gives the same 32.82.

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
CLadder and RAGTruth, train on gold only, and stage 3 (long context) is gold only. The released weights are the average
of stages 2 and 3, chosen on our own held-out panels, never on the board.

**Contamination checks.** Every training example was checked against the 0.3 suite rows (same normalized state and
question, same distinctive state, or a shared passage of ≥120 characters were dropped), and a final audit of the exact
training files compared every training question against all 563,686 questions of the 0.3 suite: 0 matches. The teacher
only labeled these training rows.
