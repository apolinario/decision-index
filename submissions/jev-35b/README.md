# JEV-35B

Self-run Decision Index 0.3 public score **59.73** (raw 69.22, breadth 58.34). Complete: 140,620 requests all ok; 140,178
scored after the official 442 exclusions. No errors, unsupported requests or abstentions. This is not the private Full
score or a maintainer-verified rank.

- Model: [autotrust/JEV-35B @ `9190ed3`](https://huggingface.co/autotrust/JEV-35B/tree/9190ed39e546dfb6173312ac9ebfbecdd8a08dbf),
  a typed-decision model on the Qwen3.5-35B-A3B architecture (35 B parameters, about 3 B active per token), bf16, one
  plain checkpoint (the decision model is merged into the weights; no LoRA at serve time).
- Results: [runs/jev-35b](https://huggingface.co/datasets/autotrust/jev-decision-index-results/tree/ce5850ff12caf28089b2b70143a14e7f328ed26f/runs/jev-35b)
  (compact `results.jsonl.gz`, `scores.json`, `index.json`, `benchmark-summary.json`, `environment.json`, `status.json`,
  the engine and driver that ran, `training-overlap-check.json`). Re-scoring `results.jsonl.gz` with
  `score --edition 0.3` reproduces 59.73.
- Kit: apolinario/decision-index @ `9eb2dbe2a358004c8782c66e40a83ac07b953fec`, unchanged scorer.
- Engine and driver: `jev_engine.py` and `fast_run.py` in this directory (the vLLM backend of `fast_run.py` ran).

| area (skill) | Knowledge & Reasoning | Language | Retrieval & Classification | Tools & Automation | Arts & Taste |
|---|---:|---:|---:|---:|---:|
| JEV-35B | 0.426 | 0.632 | 0.710 | 0.764 | 0.422 |

## Reproduce

Linux, one NVIDIA GPU with 80 GB or more (tested on one B200), Python 3.11, torch 2.13 + CUDA 13.0, transformers 5.16.1,
a vLLM development build from September 2026 (`0.1.dev1+g7e2d79176`).

```bash
hf download autotrust/JEV-35B --revision 9190ed39e546dfb6173312ac9ebfbecdd8a08dbf --local-dir JEV-35B
export JEV_LAYOUT=wide256 PYTHONPATH=submissions/jev-35b:.
python submissions/jev-35b/fast_run.py --model autotrust/JEV-35B --revision 9190ed39e546dfb6173312ac9ebfbecdd8a08dbf \
  --local-dir JEV-35B --backend vllm --suite-dir suite-0.3 --edition 0.3 --out runs/jev-35b \
  --batch 512 --max-model-len 262144 --gpu-mem 0.88
python -m decision_index score --edition 0.3 --results runs/jev-35b/results.jsonl
```

A `/v1/decide` server is also available (`bash JEV-35B/serve.sh`, from the model repository), for the latency check.

## Inference

- **Read-out.** One prefill pass per question over the model's bare-text template (`[kind]` / `[state]` / `[question]`
  / `[options]`, options labelled `A`, `B`, …, `JU`, ending `[decision]:`). The log-probabilities of the active slot
  tokens at the last position (vLLM, `logprobs_mode=processed_logprobs`, full distribution), plus the head bias from
  `adapter_vllm/decision_head.json`, divided by the per-kind temperature from `calibration.json` (noul 0.984, choice
  1.012, score 1.008; never fitted on suite rows), softmaxed. Nothing is generated; there is no abstention outcome.
- **Schema translation**, identical for every benchmark: `state`, `instructions` and option descriptions verbatim when
  strings, JSON otherwise; a choice option is rendered `"<key>: <description>"`, or `"<key>"` when the description is
  empty or equal to the key; a `noul` statement is phrased `"Is this scenario one where: <statement>?"` (already-questions
  pass verbatim).
- **Batching.** `fast_run.py` sends 512 requests at a time to one in-process vLLM engine (prefix caching on, 262,144-token
  context). `total_wall_ms` is the batch time divided by the batch size, not a per-request latency. Single-request
  latency on one B200 with `serve.sh` (one sequential client, 387 text rows sampled across the suite): median 184 ms,
  p95 345 ms.

## Capacity

Up to 256 options per choice question in one pass, every option read and none pruned; context 262,144 tokens; nothing
truncated and no request refused. Question types: `choice` and `noul`.

## Suite

Rebuilt with `suite rebuild` and imported with `suite import`; `suite verify --edition 0.3`: selected rows `b2b56d6f…`,
added rows `7429f3c9…`, GSM8K rows `75933489…`, exclusions `331df32d…`, subsets, all matching.

## Training data and overlap

TypeSafe Jev 1.13 teacher distributions (the public
[`SargeDev/jev-distill-corpus-v3`](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3)), ground-truth
decision data built from public datasets, and rows of the public
[`tasksource/tasksource-jev-typed-decisions`](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions)
(`filtered-full`, commercial licences; Decision Index benchmark sources removed upstream). The ground-truth data includes
the public **training splits** of datasets whose test splits the suite uses (among them BANKING77, CLINC150+OOS,
WinoGrande, HellaSwag, GSM8K, ContractNLI, Amazon ESCI, HoVer, RAGTruth, When2Call, New Yorker caption matching and
Habermas Machine). No test split of any suite benchmark was used; training rows sharing a 60-character sentence or a whole
state with the 0.3 public suite were removed before training; checkpoint selection and temperature fitting never used
suite data.

`training-overlap-check.json` (everything the model saw, 1,027,876 rows, against the 145,251 rows of the 0.3 suite): no
whole suite state appears in the training data; the remaining instruction and sentence matches are the fixed
per-benchmark instruction and option templates that public training-split rows share with the suite's builders.

Hardware: 1 × NVIDIA B200 (183 GB). Not affiliated with TypeSafe AI.
