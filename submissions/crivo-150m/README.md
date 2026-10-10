# Crivo-150M

[tardellirs/crivo-150m](https://huggingface.co/tardellirs/crivo-150m/tree/677d106fe58d74c324ad2ff18d0919c04113dd83) is a 149M-parameter typed-decision encoder,
fine-tuned from `cross-encoder/ettin-reranker-150m-v1`. Each question is read as one sequence
(`[CLS] instructions [SEP] state [SEP] [OPT] option …`) and every option is scored from the hidden state at its `[OPT]`
marker; a softmax over the options gives the distribution. It does not generate text.

| | Crivo-150M |
|---|---:|
| Decision Index 0.3, public index | **36.39** |
| Area skill (0.3) | Knowledge & Reasoning 11.8 · Language Understanding 41.8 · Retrieval & Classification 52.2 · Tools & Automation 60.7 · Arts & Human Taste 10.1 |
| Requests | 140,178 `ok`, 0 `unsupported`, 0 errors (`scores.json`: `"complete": true`) |
| Request time, 1× NVIDIA A100 40 GB, one request at a time | median 20 ms · mean 30 ms · p95 51 ms |
| Results | [runs/crivo-150m](https://huggingface.co/datasets/tardellirs/decision-index-results/tree/4d0016c1feb37f7609adbb6e90c0b45fc92a9736/runs/crivo-150m) (compact: no payloads, no suite text) |
| Weights | `model.safetensors` sha256 `011a550c8e4d0cb275240919a875f5e489e5c85dde5c1002608f5a804883387b` |

The run used kit `9eb2dbe` (edition 0.3) with the engine shipped in the model repository:

```sh
hf download tardellirs/crivo-150m --revision 677d106fe58d74c324ad2ff18d0919c04113dd83 --local-dir crivo-150m
cd crivo-150m
python -m decision_index pipeline --engine decision_index_engine:DecisionEngine --edition 0.3 --out runs/crivo-150m
```

The engine's defaults are the settings of this run: bf16 autocast, `batch_size=32`, `token_budget=65536`,
`max_length=32768`, **no truncation**. The encoder was trained up to 8,192 tokens; the 324 requests whose questions are
longer (almost all ToolRet requests carrying ~23k-token tool documents) are read whole with rotary positions up to 32,768
instead of being refused. A question above `max_length` would raise `Unsupported`; none exists in the 0.3 suite.

**How the run was produced.** The suite was first run with `max_length=8192`, which left 324 requests `unsupported`.
Those rows were removed from `results.jsonl` and the same run directory was resumed with `max_length=32768`; the other
140,296 rows are untouched (the runner's resume). The extended setting does not change answers on inputs up to 8,192
tokens: on 666 sampled requests (3,378 questions, all 42 benchmarks) it reproduced the 8k answers exactly (same choice
on 3,378/3,378, probability difference < 1e-4). `environment.json` records the final engine options. Running the command
above from scratch gives the same results.

Latency was measured on an A100 40 GB, not the board's RTX PRO 6000. The ~23k-token ToolRet requests are the slowest
(well under 1 s each on the A100) and do not move the median, mean or p80.

**Training data.** Bekko System One dataset v0 (all 153 subsets, capped); the *train/dev* splits of public datasets behind
the index benchmarks (ANLI, HoVer, When2Call, RAGTruth, VAST, iSarcasmEval, ACOS, NLI4CT, Amazon ESCI, ContractNLI,
BANKING77, CLINC150, HellaSwag, WinoGrande, New Yorker matching safe fold, API-Bank), rendered with the kit's own builders;
tasksource procedural-typed-decisions; ToolACE and glaive function calling converted to tool selection; CLadder variants
not used by the suite; SEntFiN entity-level financial sentiment; and synthetic smart-home cases built with the kit's row
builder and request templates (new houses only; no suite or dev case reproduced). Nothing comes from evaluation splits,
and benchmarks without a train split stay evaluation only. No Jev outputs were used.

**Contamination checks.** (1) leaf-level filter: strings ≥25 normalized characters against the rebuilt pools of all 42
benchmarks, 12,904 examples dropped; (2) builder-level check of each public-source question against the test-split output
of the same builder, 939 rows dropped; (3) exact check of every training question (~1.4M) against all 563,686 questions of
the final 0.3 suite (normalized state + instructions): 0 matches. Details in the model card.

License: CC BY-NC 4.0 (some training sources are non-commercial).
