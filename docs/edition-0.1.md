# Edition 0.1

Decision Index 0.1 (2026-09-22) is kept reproducible. Pass `--edition 0.1` to `suite`, `run`, `score`, `pipeline` and `hf-job`; the suite lives in `suite/` by default.

```sh
python -m decision_index suite rebuild --edition 0.1 --work work
python -m decision_index suite import --edition 0.1 \
    --rows work/artifacts/benchmark-suite/release-v1-rebuilt/selected-rows.jsonl.gz
python -m decision_index pipeline --edition 0.1 --engine transformers --model Qwen/Qwen2.5-7B-Instruct --out runs/qwen-7b-0.1
```

The 0.1 suite is 132,422 requests (117,764 source cases, 775,202 answer fields) over 37 benchmarks; 442 excluded questions leave 131,980 scoreable rows. Rows sha256 `750d353a3a83af615c67cfe9752e005bf09e6c28d9c4ba28d3a9f57ba8536cfd` (gzip) / `288d37207a9581187bdf83eada1983aa63de6fc50b0108e2badb229547a57f99` (uncompressed). Its headline number is the plain (raw) area mean over 19 benchmarks, not chance-corrected.

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

## Scoring and the index

`benchmark-summary.json` mirrors the lab's per-model report: accuracy, macro-F1 over observed labels, nDCG@10 for ToolRet/BRIGHT (chunk probabilities combined per query), realized quality for RouterBench, Brier for ForecastBench, case-exact accuracy for BFCL, Home appliance, SATA-Bench and ACOS, tie-accepting accuracy for ChessBench and Habermas, plus iSarcasmEval tracks.

`index.json` follows the frozen panel: per benchmark, tracks are scored with the index metric (subject/song/user/perturbation macro where the panel says so, conservative F1 with a fixed label universe and the missing-label penalty `2TP/(2TP+FP+FN+M)`), `skill = clip((raw - random)/(1 - random))`, tracks averaged with equal weight, iSarcasmEval headlined by track A English. Areas are equal-weight means of their benchmarks; the three formulas are

- `balanced_raw = 100 * mean_area(raw)` (the Decision Index)
- `balanced_skill = 100 * mean_area(skill)`
- `breadth_skill = 100 * (prod_area((0.1 + 0.9 * skill)^(1/5)) - 0.1) / 0.9`

The 25-benchmark frozen panel with interactive benchmarks at zero is also reported (`frozen_panel`) as the provisional lower bound the lab tracks, but it is not the headline number.

Parity: running `score` over the lab's own `jevfire-uncapped` results reproduces its benchmark summary and leaderboard entry exactly (55.74 / 40.86 / 39.45, all areas and per-benchmark values).

