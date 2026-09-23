# Decision Index

Reproduction kit for the **Decision Index**, a benchmark for *typed decision engines*: models that take a `state` and a set of typed `questions` (multiple-choice `choice` primitives with explicit `criteria`) and return one answer per question with a full probability distribution over the supplied options. The frozen suite has **132,422 requests** (117,764 source cases, 775,202 answer fields) over **37 benchmarks** spanning knowledge, language, retrieval, tool use and human judgment. Nineteen of them form the scored panel: every benchmark is scored with its native metric, averaged into five equal-weight areas, and the Decision Index is the mean of those five areas on a 0-100 scale (chance-normalized *skill* and *breadth* variants are computed alongside but the headline number is the plain one). Unanswered requests count as wrong, so a model that refuses long or wide inputs pays for it in the score rather than in a footnote.

This repository lets a stranger obtain the frozen suite, run any engine through it with checkpoint/resume, score every benchmark with the same metrics the leaderboard uses, compute the index, and do all of it either on a laptop or as one Hugging Face Job on an RTX PRO 6000.

Not affiliated with TypeSafe AI.

## Quickstart (local)

The frozen suite is not redistributed: several sources don't grant republishing rights, so this repository ships the recipe instead. Build it once locally (about 4 GB of downloads; accept the [HLE](https://huggingface.co/datasets/cais/hle) terms on the Hub and be logged in first):

```sh
pip install -e ".[transformers,rebuild]"
export HF_HUB_DISABLE_XET=1

python -m decision_index suite rebuild --work work
python -m decision_index suite import --dir suite \
    --rows work/artifacts/benchmark-suite/release-v1-rebuilt/selected-rows.jsonl.gz \
    --exclusions hub/excluded-questions.json --manifest hub/manifest.json
```

The rebuild pins every source and reproduces the frozen file byte for byte; `suite import` refuses the file if its hash does not match. Then:

```sh
python -m decision_index suite sample --dir suite --n 100 --out sample-100.jsonl.gz
python -m decision_index run --engine transformers --model Qwen/Qwen2.5-0.5B-Instruct \
    --rows sample-100.jsonl.gz --out runs/qwen-0.5b-sample
python -m decision_index score --results runs/qwen-0.5b-sample/results.jsonl --suite-dir suite
```

`score` writes `benchmark-summary.json` (native metric per benchmark on complete, supported case groups), `index.json` (coverage-adjusted panel scores, areas and the index) and `scores.json` (both combined, the file you would submit).

The full suite in one command, scored at the end:

```sh
python -m decision_index pipeline --engine transformers --model Qwen/Qwen2.5-7B-Instruct --out runs/qwen-7b
```

`run`/`pipeline` resume from an existing `results.jsonl`; rows whose status is `error` are retried, everything else is kept.

## Quickstart (Hugging Face Jobs)

One job runs steps 1-4 end to end on a single RTX PRO 6000 (96 GB) and uploads `results.jsonl.gz`, `benchmark-summary.json`, `index.json`, `scores.json`, `environment.json` and `status.json` to a dataset repo you own. The job downloads the suite from a Hub dataset, so first put your locally built copy in a **private** dataset under your own account (keep it private: the sources' terms apply):

```sh
hf auth login
python scripts/prepare_hub_upload.py --rows work/artifacts/benchmark-suite/release-v1-rebuilt/selected-rows.jsonl.gz --out hub-upload
hf repo create <you>/decision-index-suite --repo-type dataset --private
hf upload <you>/decision-index-suite hub-upload . --repo-type dataset

python -m decision_index hf-job --engine transformers --model Qwen/Qwen2.5-7B-Instruct \
    --suite-dataset <you>/decision-index-suite \
    --results-repo <you>/decision-index-results --run-name qwen-7b
```

What it does: creates `<you>/decision-index-results` (private, `--public` to change), uploads a snapshot of this package to `code/decision-index-src.tar.gz` in that repo, then submits `hf jobs run --flavor rtx-pro-6000 --timeout 72h pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime ...` with `HF_TOKEN` passed as a job secret. The job installs the snapshot, downloads the suite dataset, runs the engine, scores, and uploads to `runs/<run-name>/`. Use `--git-url` to install from a git checkout instead of a snapshot, `--dry-run` to print the job command, `--flavor`/`--image`/`--timeout` to change hardware (names from `hf jobs hardware`). A full 132k-request sweep on a 7B model is several hours; `--limit N` and `--compact` (drop payload/raw output from result rows) help while testing.

## The frozen suite

The suite is three files. It is not published as a public dataset, because not every source allows republishing; you build it with `suite rebuild` (below) and stage it with `suite import`. The code can also read it from a Hub dataset you control (`--suite-dataset`, default `SUITE_DATASET` in `decision_index/constants.py`, which is not public):

| File | Purpose | sha256 |
|---|---|---|
| `selected-rows.jsonl.gz` | 132,422 requests, one JSON object per line (see `docs/format.md`) | `750d353a3a83af615c67cfe9752e005bf09e6c28d9c4ba28d3a9f57ba8536cfd` (uncompressed `288d37207a9581187bdf83eada1983aa63de6fc50b0108e2badb229547a57f99`) |
| `excluded-questions.json` | 442 request ids dropped at scoring time for every engine (duplicate options, duplicated gold, 380 ToolRet/BRIGHT rows beyond nearly every context window, and their sibling chunks) | `331df32d4b719c7db43214d0e5d85859d39c3b2eb7d0b3812214cce150155e81` |
| `manifest.json` | per-benchmark counts, caps, selected group ids, seeds and file hashes | (any) |

`suite import --rows ... --exclusions ... --manifest ...` stages local copies and verifies the rows against the pinned hash (the gzip hash, or the uncompressed one when the file was recompressed), refusing to proceed on mismatch. `suite download --dataset <you>/<repo>` does the same from a private Hub copy. `scripts/prepare_hub_upload.py --rows /path/to/selected-rows.jsonl.gz` stages exactly these three files for such a copy (copies of `hub/excluded-questions.json` and `hub/manifest.json` are in this repo).

### Rebuilding from public sources

`python -m decision_index suite rebuild --work work` downloads every pinned source (git commits, Hub dataset revisions, direct URLs with sha256), normalizes each benchmark, applies the frozen sampling rules (seed 20260919, hash-ordered case selection, ESCI proportional strata, request budgets that keep linked cases intact) and writes `work/artifacts/benchmark-suite/release-v1-rebuilt/selected-rows.jsonl(.gz)`. The freeze step and all 37 normalizers were checked against the lab's raw sources: every normalized file and the final frozen file are byte-identical to the originals (uncompressed sha256 `288d372…`). `--compare path/to/reference.jsonl.gz` reports run-id and payload-hash differences if upstream data drifts. Per-benchmark sources, revisions, sampling rules and licence notes are in `docs/suite.md`; `pip install -e ".[rebuild]"` adds the extra dependencies (pyarrow, pandas, scipy, mido, python-chess, tiktoken). HLE is a gated dataset (accept its terms on the Hub and be logged in). The rebuild is the supported way to obtain the suite.

## Benchmarks

Area membership follows the leaderboard: the 25-benchmark frozen panel minus the six interactive environments (MiniWoB++, ScienceWorld, Boxoban, RTFM, Hanabi, Codenames), ChessBench folded into Knowledge & Reasoning, and the former language area split into Language Understanding (ContractNLI, iSarcasmEval, VAST) and Retrieval & Classification (BRIGHT, Amazon ESCI). Random baselines are the lab's frozen values (`decision_index/data/chance-baselines.json`): exact expectations where possible, Monte Carlo with standard error below 0.001 for F1 metrics. Display-only benchmarks are scored and reported but do not enter the index.

| # | Benchmark | Area | Report metric | Index metric | Cases | Requests | Sampling / subset rule | Random baseline | In index |
|---|---|---|---|---|---:|---:|---|---|---|
| 1 | BFCL | Tools & Automation | case exact accuracy | whole-case accuracy | 1,694 | 1,694 | All prepared cases retained. | 0.259 | yes |
| 2 | ToolRet | Tools & Automation | nDCG@10 | nDCG@10 | 7,961 | 8,032 | All prepared cases retained. | 0.094 | yes |
| 3 | API-Bank | Tools & Automation | accuracy | accuracy | 508 | 508 | All prepared cases retained. | 0.019 | display only |
| 4 | BANKING77 | Retrieval & Classification | macro-F1 | macro-F1 | 3,080 | 3,080 | All prepared cases retained. | 0.013 | display only |
| 5 | CLINC150+OOS | Retrieval & Classification | macro-F1 | macro-F1 | 5,500 | 5,500 | All prepared cases retained. | 0.006 | display only |
| 6 | RouterBench | Tools & Automation | selected quality (quality objective) | realized quality, 0-shot and 5-shot tracks averaged | 5,001 | 10,000 | 10000 requests within 10000 cap; complete linked cases retained. | 0.573 | yes |
| 9 | Home appliance simulator | Tools & Automation | case exact accuracy | case exact accuracy | 160 | 160 | All prepared cases retained. | 2.1e-07 | display only |
| 10 | SGD/SGD-X | Retrieval & Classification | macro-F1 | macro-F1 | 2,500 | 2,500 | 2500 seeded cases. | 0.399 | display only |
| 11 | ContractNLI | Language Understanding | macro-F1 | conservative macro-F1 | 123 | 123 | All prepared cases retained. | 0.309 | yes |
| 12 | ANLI | Language Understanding | macro-F1 | macro-F1 | 3,200 | 3,200 | All prepared cases retained. | 0.332 | display only |
| 20 | BPoMP | Arts & Human Judgment | accuracy | perturbation-macro accuracy | 811 | 5,000 | 5000 requests within 5000 cap; complete linked cases retained. | 0.500 | yes |
| 21 | Humicroedit | Arts & Human Judgment | accuracy | accuracy | 2,628 | 2,628 | All prepared cases retained. | 0.500 | yes |
| 22 | POP909-CL | Arts & Human Judgment | accuracy | song-macro accuracy | 2,000 | 2,000 | 2000 seeded cases. | 0.008 | yes |
| 23 | cfcolor | Arts & Human Judgment | accuracy | user-macro accuracy | 5,000 | 5,000 | 5000 seeded cases. | 0.500 | yes |
| 24 | MMLU | Knowledge & Reasoning | accuracy | subject-macro accuracy | 14,042 | 14,042 | All prepared cases retained. | 0.250 | yes |
| 25 | GPQA Diamond | Knowledge & Reasoning | accuracy | accuracy | 198 | 198 | All prepared cases retained. | 0.250 | yes |
| 26 | ARC-Easy | Knowledge & Reasoning | accuracy | accuracy | 2,376 | 2,376 | All prepared cases retained. | 0.250 | display only |
| 27 | ARC-Challenge | Knowledge & Reasoning | accuracy | accuracy | 1,172 | 1,172 | All prepared cases retained. | 0.250 | display only |
| 28 | WinoGrande | Knowledge & Reasoning | accuracy | accuracy | 1,267 | 1,267 | All prepared cases retained. | 0.500 | display only |
| 29 | HellaSwag | Knowledge & Reasoning | accuracy | accuracy | 10,042 | 10,042 | All prepared cases retained. | 0.250 | display only |
| 30 | GSM8K | Knowledge & Reasoning | accuracy | accuracy, 4- and 10-choice tracks averaged | 1,319 | 2,638 | All prepared cases retained. | 0.175 | yes |
| 31 | ChessBench | Knowledge & Reasoning | accuracy | best-move accuracy (ties accepted) | 5,000 | 5,000 | 5000 seeded cases. | 0.082 | yes |
| 32 | MuSR | Knowledge & Reasoning | accuracy | accuracy | 756 | 756 | All prepared cases retained. | 0.371 | display only |
| 33 | SATA-Bench | Knowledge & Reasoning | case exact accuracy | case exact accuracy | 1,650 | 1,650 | All prepared cases retained. | 0.013 | display only |
| 34 | SimpleBench | Knowledge & Reasoning | accuracy | accuracy | 10 | 10 | All prepared cases retained. | 0.167 | display only |
| 36 | BRIGHT | Retrieval & Classification | nDCG@10 | nDCG@10 | 1,384 | 1,384 | All prepared cases retained. | 0.045 | yes |
| 37 | Amazon ESCI | Retrieval & Classification | macro-F1 | conservative macro-F1 | 5,000 | 5,000 | 5,000 pairs, proportional locale × relevance strata, hash-ordered within strata. | 0.203 | yes |
| 38 | ACOS | Language Understanding | case exact accuracy | case exact accuracy | 1,399 | 5,479 | All prepared cases retained. | 7.6e-13 | display only |
| 39 | FinEntity | Language Understanding | macro-F1 | macro-F1 | 979 | 979 | All prepared cases retained. | 0.320 | display only |
| 40 | iSarcasmEval | Language Understanding | Sarcasm F1 · track A, English | conservative sarcasm F1, track A English | 4,600 | 4,600 | All prepared cases retained. | A-En 0.223, A-Ar 0.223, C 0.5 | yes |
| 41 | VAST | Language Understanding | macro-F1 | conservative macro-F1 | 3,006 | 3,006 | All prepared cases retained. | 0.333 | yes |
| 42 | NLI4CT | Language Understanding | macro-F1 | macro-F1 | 5,500 | 5,500 | All prepared cases retained. | 0.486 | display only |
| 43 | CRUXEval | Knowledge & Reasoning | accuracy | accuracy | 570 | 570 | All prepared cases retained. | 0.370 | yes |
| 44 | CLadder | Knowledge & Reasoning | accuracy | accuracy | 5,000 | 5,000 | 5000 seeded cases. | 0.500 | yes |
| 45 | HLE | Knowledge & Reasoning | accuracy | accuracy | 513 | 513 | All prepared cases retained. | 0.164 | display only |
| 48 | ForecastBench | Arts & Human Judgment | Brier (lower is better) | Brier (lower is better) | 10,139 | 10,139 | All prepared cases retained. | 0.250 | display only |
| 50 | Habermas Machine | Arts & Human Judgment | accuracy | group-preference accuracy (ties accepted) | 1,676 | 1,676 | All prepared cases retained. | 0.311 | yes |

Explainers per benchmark and track are in `decision_index/data/benchmarks.json`.

## Rules

These are encoded in the runner and engines, not left to the reader:

- **No truncation.** An engine that cannot fit a request raises `Unsupported`; the runner records the row as `unsupported` and the scorer counts it as wrong (coverage). Nothing is ever cut to fit.
- **No option filtering.** Every option in `criteria` is scored; the response must carry a probability for each one or `validate` rejects it.
- **No prompt tuning.** The reference engines use one fixed rendering for every benchmark. Per-benchmark prompts, few-shot examples or label-aware tricks are not part of the benchmark.
- **Unanswered = wrong.** The index uses the full frozen denominator: unsupported, errored, abstained and pending requests score zero before chance normalization. The per-benchmark report separately shows the native metric on the complete, supported case groups so you can see the gap.
- **Linked cases stay whole.** Multi-request cases (ToolRet/BRIGHT chunks, RouterBench tracks, ACOS groups) only count when every linked request succeeded.
- **Exclusions apply to everyone.** The 442 excluded questions are dropped for every engine alike, at scoring time; the frozen file is untouched.

## Scoring and the index

`benchmark-summary.json` mirrors the lab's per-model report: accuracy, macro-F1 over observed labels, nDCG@10 for ToolRet/BRIGHT (chunk probabilities combined per query), realized quality for RouterBench, Brier for ForecastBench, case-exact accuracy for BFCL, Home appliance, SATA-Bench and ACOS, tie-accepting accuracy for ChessBench and Habermas, plus iSarcasmEval tracks.

`index.json` follows the frozen panel: per benchmark, tracks are scored with the index metric (subject/song/user/perturbation macro where the panel says so, conservative F1 with a fixed label universe and the missing-label penalty `2TP/(2TP+FP+FN+M)`), `skill = clip((raw - random)/(1 - random))`, tracks averaged with equal weight, iSarcasmEval headlined by track A English. Areas are equal-weight means of their benchmarks; the three formulas are

- `balanced_raw = 100 * mean_area(raw)` (the Decision Index)
- `balanced_skill = 100 * mean_area(skill)`
- `breadth_skill = 100 * (prod_area((0.1 + 0.9 * skill)^(1/5)) - 0.1) / 0.9`

The 25-benchmark frozen panel with interactive benchmarks at zero is also reported (`frozen_panel`) as the provisional lower bound the lab tracks, but it is not the headline number.

Parity: running `score` over the lab's own `jevfire-uncapped` results reproduces its benchmark summary and leaderboard entry exactly (55.74 / 40.86 / 39.45, all areas and per-benchmark values).

## Adding an engine

Subclass `decision_index.engines.Engine` and implement `__call__(state, questions) -> (response, raw)`, where `response = {"model": ..., "answers": {key: {"type": "choice", "choice": <key>, "probabilities": {<key>: p, ...}}}}` and `raw` is whatever native output you want kept in `results.jsonl`. Raise `Unsupported(reason)` for declared capacity limits (context window, option count) and let everything else propagate as an error. Set `provenance` (dict recorded in `environment.json`) and `latency` (one sentence on what the timing covers). Register it in `decision_index/engines/__init__.py` or pass `--engine my_module:MyEngine`. See `docs/engines.md` for the two reference engines:

- `http`: a client for a Jev-compatible `POST /v1/systemone` endpoint (`--option base_url=...`, token from `DECISION_INDEX_API_KEY`).
- `transformers`: a stock causal LM scoring every option in one forward pass with a block-diagonal attention mask over a shared prompt (`--model`, `--option device=cuda|mps|cpu`, `--option attn=sdpa|eager`, `--option max_tokens=N` to declare a smaller capacity).

## Submitting a model to the leaderboard

1. Run the full suite (`pipeline` or `hf-job`) and upload the run directory to a Hub dataset (`--upload <you>/<repo>`; the job does this for you).
2. Open a pull request against this repository adding a line to `submissions/README.md` (create it if needed) with the model name, the results dataset link (`runs/<name>/scores.json` must be present), the engine/commit used and hardware. Runs must be complete (`scores.json` says `"complete": true`) and untouched: the results file is re-scored on review.
3. Mention any declared capacity limits; they are visible in `environment.json` and in the unsupported counts and are fine, as long as nothing was truncated.

## Layout

```
decision_index/
  constants.py          dataset id, hashes, panel constants
  cli.py                suite | run | score | pipeline | hf-job
  runner.py             checkpoint/resume loop, results.jsonl rows
  pipeline.py           score + index + upload
  hf_job.py             one-job submitter (rtx-pro-6000)
  engines/              Engine base, http, transformers, random
  scoring/              metrics, per-benchmark report, index math
  suite/                download, verify, sample, rebuild/
  data/                 index panel, chance baselines, benchmark catalog
hub/                    excluded-questions.json and manifest.json to upload with the rows
scripts/prepare_hub_upload.py
docs/                   suite.md (sources, licences, sampling), format.md, engines.md
tests/                  metrics, index math, report, runner
```

## Licence notes

Code in this repository is MIT. The frozen suite contains text from 36 upstream datasets plus one purpose-built benchmark; each keeps its own licence and terms (listed per benchmark in `docs/suite.md`). Several sources are research-only or non-commercial (ANLI is CC BY-NC 4.0, SGD is CC BY-SA 4.0), GPQA asks that its questions not be posted in plain text, and HLE is gated on the Hub. Treat the suite as evaluation data under those terms; do not train on it.

Not affiliated with TypeSafe AI.
