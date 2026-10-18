# Crivo-17M

[tardellirs/crivo-17m](https://huggingface.co/tardellirs/crivo-17m) is a 17M-parameter typed-decision encoder (7 layers,
hidden 256), fine-tuned from `cross-encoder/ettin-reranker-17m-v1` with knowledge distillation from Crivo-150M (#127),
on the training data of Crivo-400M (#167). It is the smallest of the Crivo family, below Crivo-32M (#131). Each question
is read as one sequence (`[CLS] instructions [SEP] state [SEP] [OPT] option …`) and every option is scored from the
hidden state at its `[OPT]` marker; a softmax over the options gives the distribution. It does not generate text.

| | Crivo-17M |
|---|---:|
| Decision Index 0.3, public index | **22.15** (raw 41.31) |
| Area skill (0.3) | Knowledge & Reasoning 1.2 · Language Understanding 21.0 · Retrieval & Classification 40.1 · Tools & Automation 36.5 · Arts & Human Taste 17.0 |
| Requests | 140,178 `ok`, 0 `unsupported`, 0 errors (`scores.json`: `"complete": true`) |
| Request time, 1× NVIDIA A100 40 GB, one request at a time | median 7.7 ms · mean 9.7 ms · p95 12.4 ms |
| Results | [runs/crivo-17m](https://huggingface.co/datasets/tardellirs/decision-index-results/tree/7ba8f95cf5900e430ef09951b7cbd1eece741903/runs/crivo-17m) (compact: no payloads, no suite text) |
| Weights | `model.safetensors` sha256 `f7fa05c95a905b140bdba722ba69297c48b8ed09737271dd950921ecea783346` |

The run used kit `9eb2dbe` (edition 0.3), in one pass, with the engine shipped in the model repository:

```sh
hf download tardellirs/crivo-17m --revision 2b97905e8d35b0afbbdfa63d6dfc92d6c3450c25 --local-dir crivo-17m
cd crivo-17m
python -m decision_index pipeline --engine decision_index_engine:DecisionEngine --edition 0.3 --out runs/crivo-17m
```

The engine's defaults are the settings of this run: one option order (`tta=1`), fp32 scoring head, SDPA attention, bf16
autocast, `batch_size=32`, `token_budget=65536`, `max_length=32768`, **no truncation**. The encoder was trained up to
8,192 tokens; longer questions (ToolRet tool documents of ~23k tokens) are read whole with rotary positions up to 32,768
instead of being refused. A question above `max_length` would raise `Unsupported`; none exists in the 0.3 suite.
The engine is the same file as Crivo-400M's. Latency was measured on an A100 40 GB, not the board's RTX PRO 6000. The
machine's Python was 3.10, so `hashlib.file_digest` in `suite/io.py` was replaced by the equivalent chunked sha256 for
that run; nothing else in the kit was changed. The uploaded `results.jsonl.gz` drops `payload` and `raw_output` (as
`--compact` does); rescored with the kit it gives the same 22.15.

**Training data.** Exactly the training files of Crivo-400M: Bekko System One dataset v0, code-generated decision
families with exact labels, format-invariance augmentations, tasksource-jev-typed-decisions and procedural typed
decisions, sibling sources of the suite's skills (argument quality, SEntFiN, RAGBench, Hermes function calling, xLAM
irrelevance, bitext intents, CUAD), train/dev splits of public datasets behind the index benchmarks (original layout and
rewritten), open teacher labels from Mapika/decider (open Qwen models), CFPB complaints, CommitPackFT, ToolACE/glaive
tool selection, CLadder variants not used by the suite, synthetic smart-home cases, and synthetic "which API next"
dialogues built from Hermes function calling. The API-Bank train split was not used, because it repeats dialogues and
catalogs of the test pool. No Jev outputs were used, and nothing comes from evaluation splits.

**Distillation.** Three students were trained with the same data and hyperparameters, one per teacher (Crivo-400M,
Crivo-68M and Crivo-150M); the Crivo-150M arm was best on our own held-out panels and is the one submitted. In stages 1
and 2 the loss mixes the gold label and the teacher's logits on the same training rows (α = 0.5); sources where the
teacher agrees with the gold label on fewer than 60% of their rows, plus smart-home control, CLadder and RAGTruth, train
on gold only, and stage 3 (long context) is gold only. The released weights are the end of stage 3. No checkpoint or arm
was chosen on the board.

**Contamination checks.** Every training example was checked against the 0.3 suite rows (same normalized state and
question, same distinctive state, or a shared passage of ≥120 characters were dropped), and a final audit of the exact
training files compared every training question against all 563,686 questions of the 0.3 suite: 0 matches. The teachers
only labeled these training rows.
