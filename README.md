# Decision Index

Reproduction kit for the **Decision Index**, a benchmark for *typed decision engines*: models that take a `state` and a set of typed `questions` (multiple-choice `choice` or yes/no `noul` primitives with explicit `criteria`) and return one answer per question with a probability for every supplied option. It lets anyone rebuild the frozen suite from public sources, run any engine through it with checkpoint/resume, score every benchmark with the leaderboard's own scorers, and compute the index, on a laptop or as one Hugging Face Job.

The live board is **Decision Index 0.2** (2026-09-24), the default everywhere in this kit. It averages 40 benchmarks in five equal-weight areas, each chance-corrected (0 = random guessing, 100 = perfect) and coverage-adjusted (unanswered = wrong). Edition 0.1 stays reproducible with `--edition 0.1` ([docs/edition-0.1.md](docs/edition-0.1.md)).

Not affiliated with TypeSafe AI.

## What changed in 0.2

These are the site's own summary lines:

- Scores are now chance-corrected: 0 means random guessing and 100 means perfect, so yes/no questions no longer give free points.
- The index now counts 40 benchmarks, up from 19: 7 brand-new ones (MMLU-Pro, BBH, When2Call, RAGTruth, HoVer, New Yorker caption matching, PhishNChips), plus 15 we already ran that now count, such as HLE, ANLI, WinoGrande, HellaSwag, BANKING77 and CLINC150.
- MMLU left the index: top models have nearly maxed it out, and MMLU-Pro takes its place. ARC-Easy, ARC-Challenge and SimpleBench are still shown but don't count.
- ToolRet and BRIGHT are run on a subset of their data that doesn't skew results but is faster and cheaper to run.
- 15 new models on the board, plus newer versions of 14 existing ones.

In the kit this means: the 0.2 suite is the 0.1 rows with ToolRet and BRIGHT cut to stratified subsets of 1,000 and 550 queries (121,057 requests after ACOS is cut to its 400-review subset at scoring time; 120,615 scoreable after the same 442 exclusions), plus 30,419 requests for the seven new benchmarks. The raw (not chance-corrected) index is still computed and reported as a secondary score.

## Quickstart (local)

The suite is not redistributed: several sources don't allow republishing (RAGTruth's Yelp and MS MARCO contexts, for one), so this repository ships the recipe and you build the files once. About 7 GB of downloads (the HoVer Wikipedia database alone is 2.2 GB) and 17 GB of working space; accept the [HLE](https://huggingface.co/datasets/cais/hle) terms on the Hub and be logged in first.

```sh
pip install -e ".[transformers,rebuild]"
export HF_HUB_DISABLE_XET=1

python -m decision_index suite rebuild --work work
python -m decision_index suite import \
    --rows work/artifacts/benchmark-suite/release-v2-rebuilt/selected-rows.jsonl.gz \
    --added-rows work/artifacts/benchmark-suite/release-v2-rebuilt/added-rows.jsonl.gz
```

`suite rebuild` rebuilds the 0.1 rows, cuts them to 0.2 with the published subset lists, and builds the seven new benchmarks; every source is pinned and both files come out byte-identical to the lab's. `suite import` stages them in `suite-0.2/` with the exclusion list and manifest from `hub/`, and refuses files whose hash does not match. Then:

```sh
python -m decision_index suite sample --n 100 --out sample-100.jsonl.gz
python -m decision_index run --engine transformers --model Qwen/Qwen2.5-0.5B-Instruct \
    --rows sample-100.jsonl.gz --out runs/qwen-0.5b-sample
python -m decision_index score --results runs/qwen-0.5b-sample/results.jsonl
```

The full suite in one command, scored at the end, entirely on your own machine (nothing is uploaded unless you pass `--upload`):

```sh
python -m decision_index pipeline --engine transformers --model Qwen/Qwen2.5-7B-Instruct --out runs/qwen-7b
```

For a model behind your own server that speaks the `/v1/systemone` wire format, use the `http` engine:

```sh
python -m decision_index pipeline --engine http --option base_url=http://127.0.0.1:8000 --option model=my-model --out runs/my-model
```

`run`/`pipeline` resume from an existing `results.jsonl`; rows whose status is `error` are retried, everything else is kept. A 0.1 `results.jsonl` can be rescored under 0.2 (the 0.2 base rows are a subset of 0.1), but the seven new benchmarks still have to be run; resuming the same output directory under 0.2 runs only those.

## Quickstart (Hugging Face Jobs)

One job runs everything on a single RTX PRO 6000 (96 GB) and uploads `results.jsonl.gz`, `benchmark-summary.json`, `index.json`, `scores.json`, `environment.json` and `status.json` to a dataset repo you own. The job reads the suite from a Hub dataset, so first put your locally built copy in a **private** dataset under your account (keep it private: the sources' terms apply):

```sh
hf auth login
python scripts/prepare_hub_upload.py \
    --rows work/artifacts/benchmark-suite/release-v2-rebuilt/selected-rows.jsonl.gz \
    --added-rows work/artifacts/benchmark-suite/release-v2-rebuilt/added-rows.jsonl.gz --out hub-upload
hf repo create <you>/decision-index-suite-0.2 --repo-type dataset --private
hf upload <you>/decision-index-suite-0.2 hub-upload . --repo-type dataset

python -m decision_index hf-job --engine transformers --model Qwen/Qwen2.5-7B-Instruct \
    --suite-dataset <you>/decision-index-suite-0.2 \
    --results-repo <you>/decision-index-results --run-name qwen-7b
```

`hf-job` creates `<you>/decision-index-results` (private unless `--public`), uploads a snapshot of this package to `code/decision-index-src.tar.gz`, and submits `hf jobs run --flavor rtx-pro-6000 --timeout 72h pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime ...` with `HF_TOKEN` as a job secret. The job installs the snapshot, downloads the suite, runs the engine, scores, and uploads to `runs/<run-name>/`. `--git-url` installs from a git checkout instead, `--dry-run` prints the command, `--flavor`/`--image`/`--timeout` change the hardware, `--limit N` and `--compact` help while testing, `--edition 0.1` runs the old edition.

## Scoring and the index

`score` (and the end of `pipeline`) writes three files:

- `benchmark-summary.json`: the native metric of every benchmark on its complete, supported case groups (accuracy, macro-F1, nDCG@10, RouterBench quality, Brier, case-exact accuracy, tie-accepting accuracy; F1 on the hallucinated class for RAGTruth; PhishNChips on its `verdict` question only), with answered/unsupported/error counts.
- `index.json`: per benchmark `raw`, `skill`, `coverage` and chance level; the five areas; `index` (the Decision Index) and `raw_index`.
- `scores.json`: both combined, with completion flags. This is the file a submission points at.

The 0.2 index, exactly as the board computes it (`decision_index/scoring/index02.py`, panel and chance levels in `decision_index/data/index-0.2.json`):

1. **Per benchmark, coverage first.** Unanswered, unsupported, errored and abstained requests count as wrong: `raw = native score × answered / requests`. The 18 benchmarks carried over from the 0.1 panel keep their track-level scoring (unanswered groups score zero inside the metric).
2. **Chance correction.** `skill = clip((raw − chance) / (1 − chance), 0, 1)`. Chance is per track for GSM8K and RouterBench, per query (expected nDCG@10 of a random ranking) for ToolRet and BRIGHT, F1 of random guessing for the F1 benchmarks, and the mean of 1/options otherwise. **ForecastBench** enters against its baseline instead: `clip((0.25 − Brier) / 0.25) × coverage`, so always predicting 0.5 scores zero.
3. **Areas and index.** Each area is the plain mean of its benchmarks; the index is `100 × mean of the five areas`. `raw_index` is the same with `raw` in place of `skill`. MMLU, ARC-Easy, ARC-Challenge and SimpleBench are scored and shown but not counted. The board treats scores within 0.25 index points of the next one as tied (`index02.ranks`).

Parity: `score` over the lab's own results reproduces every one of the 49 entrants on the live 0.2 board, index, raw index, breadth, all five areas and all 40 per-benchmark values. `tests/test_index02.py` checks the index math against the published per-benchmark results of Jev (51.67), AutoJev-27B (50.94), Hopper (30.01) and Verdict (1.82).

## The suite

| File | Contents | sha256 |
|---|---|---|
| `selected-rows.jsonl.gz` | 124,971 requests: the 0.1 rows minus ToolRet/BRIGHT queries outside their subsets (excluded ToolRet/BRIGHT rows are carried over unchanged and never scored) | uncompressed `b2b56d6fb636837ca469e689087bdbf373dda8de7638aa2da6793e6eda0792d5` (the lab's gzip is `25aac5e890a54a3172c7a0c184b4cc8b9a43f10b6ee89bbad8da923be423c656`) |
| `added-rows.jsonl.gz` | 30,419 requests of the seven new benchmarks | uncompressed `7429f3c9cdddb772c1cfc42bb2a45e8516b0032152b746e6929f1c8b52f4ce89` |
| `excluded-questions.json` | 442 request ids dropped at scoring time for every engine (unchanged from 0.1) | `331df32d4b719c7db43214d0e5d85859d39c3b2eb7d0b3812214cce150155e81` |
| `manifest.json` | counts, subset rules, sources and licences (`hub/0.2/manifest.json`) | (any) |

The ToolRet/BRIGHT subset lists and the ACOS 400-review subset ship with the package (`decision_index/data/release-v2/`). Rebuilt gzip files differ from the lab's in their header, so the kit verifies the uncompressed hash. Sources, revisions, sampling and licences per benchmark: [docs/suite.md](docs/suite.md). Row and result formats: [docs/format.md](docs/format.md).

## Benchmarks (0.2)

| # | Benchmark | Area | Metric | Chance | Requests | |
|---|---|---|---|---|---:|---|
| 1 | BFCL | Tools & Automation | case exact accuracy | 0.2592 | 1,694 | |
| 2 | ToolRet | Tools & Automation | nDCG@10 | per query (0.0937) | 1,000 | subset |
| 3 | API-Bank | Tools & Automation | accuracy | 0.0189 | 508 | |
| 4 | BANKING77 | Retrieval & Classification | macro-F1 | 0.0127 | 3,080 | |
| 5 | CLINC150+OOS | Retrieval & Classification | macro-F1 | 0.006 | 5,500 | |
| 6 | RouterBench | Tools & Automation | selected quality | per track (0.5735) | 10,000 | |
| 9 | Home appliances | Tools & Automation | case exact accuracy | 0 | 160 | |
| 10 | SGD | Retrieval & Classification | macro-F1 | 0.399 | 2,500 | |
| 11 | ContractNLI | Language Understanding | conservative macro-F1 | 0.3085 | 123 | |
| 12 | ANLI | Language Understanding | macro-F1 | 0.3324 | 3,200 | |
| 20 | BPoMP | Arts & Human Taste | accuracy | 0.5 | 5,000 | |
| 21 | Humicroedit | Arts & Human Taste | accuracy | 0.5 | 2,628 | |
| 22 | POP909 | Arts & Human Taste | song-macro accuracy | 0.0078 | 2,000 | |
| 23 | cfcolor | Arts & Human Taste | user-macro accuracy | 0.5 | 5,000 | |
| 24 | MMLU | shown, not counted | accuracy | 0.25 | 14,033 | |
| 25 | GPQA Diamond | Knowledge & Reasoning | accuracy | 0.25 | 196 | |
| 26 | ARC-Easy | shown, not counted | accuracy | 0.2502 | 2,376 | |
| 27 | ARC-Challenge | shown, not counted | accuracy | 0.2502 | 1,172 | |
| 28 | WinoGrande | Language Understanding | accuracy | 0.5 | 1,267 | |
| 29 | HellaSwag | Language Understanding | accuracy | 0.25 | 10,042 | |
| 30 | GSM8K | Knowledge & Reasoning | accuracy | per track (0.25, 0.1) | 2,638 | |
| 31 | ChessBench | Knowledge & Reasoning | best-move accuracy | 0.0819 | 5,000 | |
| 32 | MuSR | Knowledge & Reasoning | accuracy | 0.371 | 752 | |
| 33 | SATA-Bench | Knowledge & Reasoning | case exact accuracy | 0.0131 | 1,650 | |
| 34 | SimpleBench | shown, not counted | accuracy | 0.1667 | 10 | |
| 36 | BRIGHT | Retrieval & Classification | nDCG@10 | per query (0.0448) | 550 | subset |
| 37 | Amazon ESCI | Retrieval & Classification | conservative macro-F1 | 0.2027 | 5,000 | |
| 38 | ACOS | Language Understanding | case exact accuracy | 0 | 1,565 | subset |
| 39 | FinEntity | Language Understanding | macro-F1 | 0.3201 | 979 | |
| 40 | iSarcasmEval | Language Understanding | sarcasm F1, track A English | 0.2227 | 4,600 | |
| 41 | VAST | Language Understanding | conservative macro-F1 | 0.3333 | 3,006 | |
| 42 | NLI4CT | Language Understanding | macro-F1 | 0.486 | 5,500 | |
| 43 | CRUXEval | Knowledge & Reasoning | accuracy | 0.3697 | 570 | |
| 44 | CLadder | Knowledge & Reasoning | accuracy | 0.5 | 5,000 | |
| 45 | HLE | Knowledge & Reasoning | accuracy | 0.1641 | 501 | |
| 48 | ForecastBench | Arts & Human Taste | Brier, against the 0.25 baseline | 0.25 Brier | 10,139 | |
| 50 | Habermas Machine | Arts & Human Taste | group-preference accuracy | 0.311 | 1,676 | |
| 56 | PhishNChips | Retrieval & Classification | accuracy (verdict) | 0.5 | 2,000 | new |
| 57 | MMLU-Pro | Knowledge & Reasoning | accuracy | 0.1109 | 12,032 | new |
| 58 | BBH | Knowledge & Reasoning | accuracy | 0.3101 | 5,507 | new |
| 59 | RAGTruth | Language Understanding | F1 on hallucinated class | 0.4113 | 2,700 | new |
| 61 | HoVer | Retrieval & Classification | accuracy | 0.5 | 4,000 | new |
| 62 | When2Call | Tools & Automation | accuracy | 0.25 | 3,652 | new |
| 64 | New Yorker | Arts & Human Taste | accuracy | 0.2 | 528 | new |

Requests are after the 442 exclusions. The six interactive environments of the original panel (MiniWoB++, ScienceWorld, Boxoban, RTFM, Hanabi, Codenames) are still unrun and stay out.

## Rules

Encoded in the runner and engines, not left to the reader:

- **No truncation.** An engine that cannot fit a request raises `Unsupported`; the row is recorded as `unsupported` and counts as wrong. Nothing is cut to fit.
- **No option filtering.** Every option in `criteria` gets a probability or `validate` rejects the response.
- **No prompt tuning.** The reference engines use one fixed rendering for every benchmark.
- **Unanswered = wrong.** Unsupported, errored, abstained and pending requests score zero before chance correction. The per-benchmark report shows the native metric on answered cases separately so you can see the gap.
- **Linked cases stay whole.** Multi-request cases (ToolRet/BRIGHT chunks, RouterBench tracks, ACOS reviews) only count when every linked request succeeded.
- **Exclusions apply to everyone,** at scoring time; the frozen files are untouched.

## Engines

Subclass `decision_index.engines.Engine`, implement `__call__(state, questions) -> (response, raw)`, raise `Unsupported(reason)` for declared capacity limits, and pass `--engine my_module:MyEngine`. The kit ships two reference engines (details in [docs/engines.md](docs/engines.md)):

- `http`: a client for any `POST /v1/systemone` server (`--option base_url=...`, `--option model=...`, extra request fields via `--option extra='{...}'`, token from `DECISION_INDEX_API_KEY`).
- `transformers`: a stock causal LM scoring every option in one forward pass over a shared prompt.

The board's entrants were run with their authors' own inference code. Entrants whose authors publish a `/v1/systemone` server can be driven by `http` once that server is up; the others need their own harness. [docs/engines.md](docs/engines.md) lists which is which.

## Submitting a model to the leaderboard

1. Run the full 0.2 suite (`pipeline` or `hf-job`) and upload the run directory to a Hub dataset (`--upload <you>/<repo>`; the job does this for you).
2. Open a pull request adding a line to `submissions/README.md` (create it if needed) with the model name, the results dataset link (`runs/<name>/scores.json` must be present), the engine/commit used and hardware. Runs must be complete (`scores.json` says `"complete": true`) and untouched: the results file is re-scored on review.
3. Mention any declared capacity limits; they show in `environment.json` and in the unsupported counts and are fine, as long as nothing was truncated.

## Layout

```
decision_index/
  editions.py           0.1 and 0.2: hashes, counts, subsets
  cli.py                suite | run | score | pipeline | hf-job
  runner.py             checkpoint/resume loop, results.jsonl rows
  pipeline.py           score + index + upload
  hf_job.py             one-job submitter (rtx-pro-6000)
  engines/              Engine base, http, transformers, random
  scoring/              metrics, per-benchmark report, 0.1 index, 0.2 index (index02), new-benchmark scorer (added)
  suite/                download, verify, sample, rebuild/ (0.1 normalizers, 0.2 cut, new benchmarks)
  data/                 panels, chance levels, benchmark catalog, release-v2 subset lists
hub/                    exclusions and manifests to stage with the rows (hub/0.2 for 0.2)
scripts/prepare_hub_upload.py
docs/                   suite.md, format.md, engines.md, edition-0.1.md
tests/                  metrics, index math (0.1 and 0.2), editions, report, runner
```

## Licence notes

Code in this repository is MIT. The suite contains text from 43 upstream datasets plus one purpose-built benchmark, each under its own terms (listed per benchmark in [docs/suite.md](docs/suite.md)). Several are research-only or non-commercial (ANLI CC BY-NC 4.0; RAGTruth's MS MARCO contexts), some forbid redistribution (RAGTruth's Yelp contexts), some are share-alike (SGD, HoVer), GPQA asks that its questions not be posted in plain text, BBH carries the BIG-bench canary, and HLE is gated. Treat the suite as evaluation data under those terms; do not train on it and do not republish it.

Not affiliated with TypeSafe AI.
