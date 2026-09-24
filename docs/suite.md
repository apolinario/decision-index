# The frozen suite: sources, sampling and licences

## Edition 0.2

Edition 0.2 is built from edition 0.1 plus seven new benchmarks; nothing in the 0.1 rows is regenerated or edited.

- **ToolRet and BRIGHT subsets.** `selected-rows.jsonl` keeps every 0.1 line except the ToolRet (2) and BRIGHT (36) rows whose query group is outside `decision_index/data/release-v2/retrieval-subsets.json`. The subsets are stratified by `metadata.domain` with proportional allocation (at least one query per domain), sampled without replacement per domain with `random.Random(20260922 + catalog_id)`: 1,000 of 7,704 ToolRet queries and 550 of 1,297 BRIGHT queries (queries not already excluded). Kept queries keep all their candidate chunks. Rows of excluded questions are carried over unchanged (they are never scored), so the file has 124,971 lines, 1,328 ToolRet and 637 BRIGHT.
- **ACOS subset.** ACOS stays in the rows file but is scored on 400 of its 1,399 reviews (1,565 of 5,479 rows), stratified by domain (Laptop 233, Restaurant 167) with `random.Random(20260960)`; the kept run ids are in `decision_index/data/release-v2/acos-subset.json` and `run`, `sample` and `score` skip the others. That leaves 121,057 requests, 120,615 after the same 442 exclusions as 0.1.
- **Seven new benchmarks** in `added-rows.jsonl` (30,419 requests), built by `decision_index/suite/build/adapters_added.py` from pinned sources. The questions, options, labels and option order are the published ones; wording is mechanical and identical for every entrant. Run ids keep the lab's prefixes (`candidates-v3:` and `calibration-v1:`).

| # | Benchmark | Requests | Source (pinned) | Framing | Licence note |
|---|---|---:|---|---|---|
| 57 | MMLU-Pro | 12,032 | TIGER-Lab/MMLU-Pro @b189ec7 (Hub): data/test parquet (sha256 pinned) | question as state; letters A.. for the published options (3 to 10) | MIT |
| 58 | BBH | 5,507 | suzgunmirac/BIG-Bench-Hard @9ee07bd, bbh/*.json | 23 fixed-option task files (the four free-form tasks dropped); input verbatim as state; options parsed from its own `Options:` block, or the task's fixed answer set; four malformed source items skipped | MIT; carries the BIG-bench canary, never train on it |
| 59 | RAGTruth | 2,700 | ParticleMedia/RAGTruth @c103204, dataset/{response,source_info}.jsonl (sha256 pinned) | all test responses; state `{prompt, response}`; one yes/no question "contains content not supported by the context"; gold = any annotated span; scored as F1 on the hallucinated class | MIT annotations; contexts from CNN/Daily Mail, MS MARCO (non-commercial) and the Yelp Open Dataset (no redistribution): rebuild locally, never republish the rows |
| 61 | HoVer | 4,000 | hover-nlp/hover @39b8469, data/hover/hover_dev_release_v1.1.json; nlp.cs.unc.edu/data/hover/wiki_wo_links.db (2.2 GB, sha256 pinned) | all dev claims; state = the claim plus the full introduction of every document named in `supporting_facts` (oracle documents); SUPPORTED / NOT_SUPPORTED | CC BY-SA 4.0 (data), Wikipedia text CC BY-SA |
| 62 | When2Call | 3,652 | nvidia/When2Call @0582f77 (Hub): test/when2call_test_mcq.jsonl (sha256 pinned) | state `{tools, question}`; A..D = direct, tool call, ask for information, cannot answer, in published order | CC BY 4.0 |
| 64 | New Yorker | 528 | jmhessel/newyorker_caption_contest @d81cbab (Hub): matching/test parquet (sha256 pinned) | text only (scene, descriptions, entities); A..E = the five candidate captions | CC BY 4.0 annotations; no cartoon images used |
| 56 | PhishNChips | 2,000 | AreLit/PhishNChips @89afcc3 (Hub): core_emails.csv (sha256 pinned); questions from anisselbd/jev-phishing-bench @1d56e8c, run_jev.py | original nine-question payload, records in `random.Random(20260916)` order; only `verdict` is scored (the five signal questions have no gold) | per the dataset's SOURCE_LICENSES.md (project content MIT; URL sources cleared for academic use with attribution) |

Chance for the new benchmarks is the mean of 1/options over the scored fields; for RAGTruth it is the F1 of a fair coin, `p / (p + 0.5)` with `p` the share of hallucinated responses (943/2,700), 0.4113.

Rebuilding 0.2 (`suite rebuild --edition 0.2`) runs the 0.1 rebuild below, applies the subset cut, then downloads and builds the new benchmarks (about 2.4 GB more, mostly the HoVer database) into `work/artifacts/benchmark-suite/release-v2-rebuilt/`. The cut was checked against the lab's 0.1 file and the new-benchmark builders against a fresh download of every pinned source: both outputs are byte-identical to the lab's (uncompressed sha256 `b2b56d6f…` and `7429f3c9…`). `--only 57 58` rebuilds only some of the new benchmarks (the output is then partial and fails the hash check). The builders need `pyarrow>=20`: the MMLU-Pro parquet uses page-index statistics older versions cannot read.

## Edition 0.1

Every benchmark below was normalized from its pinned source by `decision_index/suite/build/` and then frozen by `freeze.py` (seed 20260919). Case selection is a deterministic hash order `sha256("20260919:<catalog_id>:<group_id>")`: capped benchmarks keep the lowest hashes up to the cap; request-budget benchmarks add whole linked cases in hash order while they fit; Amazon ESCI allocates its 5,000 pairs proportionally over locale × relevance strata (largest remainder) and takes the lowest hashes inside each stratum. Everything else keeps all prepared cases. The exclusion manifest (442 request ids) is applied at scoring time only.

All 37 normalizers and the freeze step were run against the lab's pinned raw sources and produced byte-identical files. If an upstream source changes after its pinned revision, the rebuilt file will differ; `suite rebuild --compare` reports which run ids or payload hashes moved.

Licence notes are what the pinned sources state (or "see repository LICENSE" when only a licence file is present, and "not stated" when none was found). Verify before redistributing; none of these terms transfer to this repository's MIT licence.

| # | Benchmark | Requests | Source (pinned) | Sampling / subset rule | Adaptation | Licence note |
|---|---|---:|---|---|---|---|
| 1 | BFCL | 1,694 | gorilla-llm/gorilla (GitHub) @916260d; data/BFCL_v3_{simple,live_simple,live_multiple}.json + possible_answer | All prepared cases retained. | B mechanical: Published tool schemas | Apache-2.0 (repo LICENSE) |
| 2 | ToolRet | 8,032 | mangopy/ToolRet-Queries @b8c76ad and mangopy/ToolRet-Tools @e06c38c (Hub datasets); mangopy/tool-retrieval-benchmark @c4181d9 for the task-to-category map | All prepared cases retained. | B mechanical: Query-only search over published tool corpus | see repository LICENSE; queries/tools aggregate many upstream tool datasets with their own terms |
| 3 | API-Bank | 508 | AlibabaResearch/DAMO-ConvAI @188835d, api-bank/ (apis/*.py catalog, lv1-lv2-samples) | All prepared cases retained. | B mechanical: Published API catalog | see api-bank/LICENSE in the repo |
| 4 | BANKING77 | 3,080 | PolyAI-LDN/task-specific-datasets (GitHub, master) banking_data/test.csv, categories.json, sha256 pinned (the PolyAI/banking77 Hub repo is a loading script that downloads these) | All prepared cases retained. | A direct: Original77 intent labels | CC BY 4.0 |
| 5 | CLINC150+OOS | 5,500 | clinc/oos-eval (GitHub, master) data/data_full.json, sha256 pinned | All prepared cases retained. | A direct: Original150 intents plus OOS | CC BY 3.0 (per dataset card) |
| 6 | RouterBench | 10,000 | withmartian/routerbench @7840214 (Hub): routerbench_0shot.pkl, routerbench_5shot.pkl | 10000 requests within 10000 cap; complete linked cases retained. | B mechanical: Original11 candidate models | see repository LICENSE |
| 9 | Home appliance simulator | 160 | generated by decision_index/suite/build/home_appliance.py (seed 2026091807); no download | All prepared cases retained. | C semantic options: Locally authored device/command scenarios | MIT (this repository) |
| 10 | SGD/SGD-X | 2,500 | google-research-datasets/dstc8-schema-guided-dialogue @e852981, test/ | 2500 seeded cases. | B mechanical: Original service schemas and intent labels | CC BY-SA 4.0 |
| 11 | ContractNLI | 123 | stanfordnlp/contract-nli @eced652, resources/contract-nli.zip (test.json) | All prepared cases retained. | A direct: Original hypotheses and NLI labels | see repository LICENSE and dataset page terms |
| 12 | ANLI | 3,200 | facebook/anli @8e4813d (Hub): plain_text/test_r{1,2,3}.parquet | All prepared cases retained. | A direct: Original3 NLI labels | CC BY-NC 4.0 |
| 20 | BPoMP | 5,000 | Zenodo record 7299879, BPoMP_datasets_p{1,2,3}_out_of_3.json (sha256 pinned) | 5000 requests within 5000 cap; complete linked cases retained. | B mechanical: Released original/perturbed poem pairs | per the Zenodo record |
| 21 | Humicroedit | 2,628 | cs.rochester.edu/u/nhossain/semeval-2020-task-7-dataset.zip, subtask-2/test.csv (sha256 pinned) | All prepared cases retained. | B mechanical: Released paired headline edits | SemEval-2020 Task 7 terms |
| 22 | POP909-CL | 2,000 | AndyWeasley2004/POP909-CL-Dataset @be90943, POP909_processed/*.mid | 2000 seeded cases. | C semantic options: Root-designed fixed chord vocabulary | see repository LICENSE; POP909 underlying data is research-use |
| 23 | cfcolor | 5,000 | dgp.toronto.edu/~donovan/cfcolor/cfcolor.zip (release/allMTurkRatings.mat, themeData.mat; sha256 pinned) | 5000 seeded cases. | B mechanical: Released rated palettes | per the project page |
| 24 | MMLU | 14,042 | cais/mmlu @c30699e (Hub): all/test parquet | All prepared cases retained. | A direct: Original multiple-choice options | MIT |
| 25 | GPQA Diamond | 198 | github.com/idavidrein/gpqa dataset.zip (sha256 pinned; password in the upstream README) | All prepared cases retained. | A direct: Original multiple-choice options | CC BY 4.0; authors ask not to publish the questions in plain text |
| 26 | ARC-Easy | 2,376 | allenai/ai2_arc (Hub, latest): ARC-Easy/test parquet | All prepared cases retained. | A direct: Original multiple-choice options | CC BY-SA 4.0 |
| 27 | ARC-Challenge | 1,172 | allenai/ai2_arc (Hub, latest): ARC-Challenge/test parquet | All prepared cases retained. | A direct: Original multiple-choice options | CC BY-SA 4.0 |
| 28 | WinoGrande | 1,267 | allenai/winogrande @01e7417 (Hub): winogrande_xl/validation parquet | All prepared cases retained. | A direct: Original2 completion options | see dataset card |
| 29 | HellaSwag | 10,042 | Rowan/hellaswag @218ec52 (Hub): data/validation parquet | All prepared cases retained. | A direct: Original continuation options | MIT |
| 30 | GSM8K | 2,638 | openai/gsm8k @740312a (Hub): main/test parquet; deterministic numeric distractors, seed 20260917 | All prepared cases retained. | B mechanical: Algorithmic numeric distractors | MIT |
| 31 | ChessBench | 5,000 | storage.googleapis.com/searchless_chess/data/test/action_value_data.bag (sha256 pinned); google-deepmind/searchless_chess @90ae0e6 | 5000 seeded cases. | B mechanical: All legal moves from source positions | see repository LICENSE (Apache-2.0 code) |
| 32 | MuSR | 756 | Zayne-sprague/MuSR @b1f4d41, datasets/*.json | All prepared cases retained. | A direct: Original multiple-choice options | MIT |
| 33 | SATA-Bench | 1,650 | sata-bench/sata-bench @371dd0c, sata_bench_final_2025.json | All prepared cases retained. | A direct: Original select-all alternatives | see repository LICENSE |
| 34 | SimpleBench | 10 | simple-bench/SimpleBench @fbc2e42, simple_bench_public.json (10 public items) | All prepared cases retained. | A direct: Original public multiple-choice options | MIT |
| 36 | BRIGHT | 1,384 | xlangai/BRIGHT @3066d29 (Hub): examples/*.parquet, documents/*.parquet | All prepared cases retained. | B mechanical: Query-only search over published document corpus | CC BY 4.0 |
| 37 | Amazon ESCI | 5,000 | amazon-science/esci-data @7916cdf: shopping_queries_dataset_{examples,products}.parquet (git LFS) | 5,000 pairs, proportional locale × relevance strata, hash-ordered within strata. | A direct: Original4 relevance labels | Apache-2.0 |
| 38 | ACOS | 5,479 | NUSTM/ACOS @45d179a, data/*/*_quad_*.tsv | All prepared cases retained. | B mechanical: Source categories crossed with3 source sentiment labels | not stated in the repository |
| 39 | FinEntity | 979 | yixuantt/FinEntity @3b6cedc, data/FinEntity.json | All prepared cases retained. | B mechanical: Original sentiment labels, provided source entity spans | not stated in the repository |
| 40 | iSarcasmEval | 4,600 | iabufarha/iSarcasmEval @dfc708b, test/task_{A,B,C}_{En,Ar}_test.csv | All prepared cases retained. | A direct: Original binary/multilabel/pair alternatives | see repository LICENSE |
| 41 | VAST | 3,006 | emilyallaway/zero-shot-stance @e7c4775, data/VAST/vast_test.csv | All prepared cases retained. | A direct: Original3 stance labels | not stated in the repository |
| 42 | NLI4CT | 5,500 | ai-systems/Task-2-SemEval-2024 @7f32fa6: test.json, gold_test.json, training_data.zip (trial documents) | All prepared cases retained. | A direct: Original entailment labels | SemEval-2024 Task 2 terms |
| 43 | CRUXEval | 570 | facebookresearch/cruxeval @190faf1: data/cruxeval.jsonl and samples/ (codellama-7b generations used as distractors) | All prepared cases retained. | C semantic options: Archived model-generated wrong outputs plus source gold | MIT |
| 44 | CLadder | 5,000 | causalNLP/cladder @3d2d116, data/cladder-v1.zip (balanced questions) | 5000 seeded cases. | A direct: Original yes/no outcomes | see repository LICENSE |
| 45 | HLE | 513 | cais/hle @5a81a4c (Hub, gated): data/test parquet; text-only multiple-choice items | All prepared cases retained. | B mechanical: Original multiple-choice options only | MIT (per dataset card); gated access |
| 48 | ForecastBench | 10,139 | forecastingresearch/forecastbench-datasets @da48cfb: datasets/question_sets, datasets/resolution_sets (resolutions 2026-07-01 to 2026-09-18) | All prepared cases retained. | B mechanical: Binary events and source joint-event directions | see repository LICENSE |
| 50 | Habermas Machine | 1,676 | google-deepmind/habermas_machine @7923b71; hm_all_candidate_comparisons.parquet from storage.googleapis.com/habermas_machine/datasets (sha256 pinned, 0.4 GB) | All prepared cases retained. | B mechanical: Released consensus statements | see repository LICENSE |

## Rebuild procedure

```sh
pip install -e ".[rebuild]"
python -m decision_index suite rebuild --work work                          # 0.2: the 0.1 rebuild, the cut, the new benchmarks
python -m decision_index suite rebuild --edition 0.1 --work work            # 0.1 only: download + normalize + freeze
python -m decision_index suite rebuild --edition 0.1 --work work --only 24 25 # a subset
python -m decision_index suite rebuild --edition 0.1 --work work --skip-download --compare suite/selected-rows.jsonl.gz
```

`work/` mirrors the lab layout (`artifacts/benchmark-suite/{raw,normalized,requests,active,release-v1-rebuilt,release-v2-rebuilt}` and `data/sources`) so that provenance paths inside the rows match the frozen file. Downloads: shallow git fetches at the pinned commits (sparse checkouts for the large monorepos), `snapshot_download` at pinned Hub revisions, direct URLs verified by sha256. Roughly 4.5 GB in total; ESCI (2 GB), BRIGHT (0.6 GB), the Habermas comparisons (0.4 GB) and the ToolRet tool corpus dominate. The retrieval benchmarks build SQLite FTS5 BM25 indexes over the published corpora (`retrieval-indexes-v2/`), which takes a few minutes.

Benchmark-specific notes:

- **GPQA Diamond** ships as a password-protected zip in the upstream repository; the password is public in that README and the normalizer uses it. The authors ask that questions not be posted in plain text; the Hub suite dataset contains them, so consider gating it.
- **HLE** is gated on the Hub; accept the terms and log in before rebuilding.
- **Home appliance simulator** is purpose-built: the generator in this repository is the source, so it always rebuilds exactly.
- **ForecastBench** keeps resolutions dated 2026-07-01 to 2026-09-18; a later upstream snapshot has more resolved questions and would change the file if the pinned commit were not used.
- **ToolRet / BRIGHT** candidates are query-only BM25 top-32 from the published corpora with source exclusion lists; unjudged candidates are never treated as verified negatives, and 380 over-long rows plus 35 sibling chunks are excluded at scoring time.
- **RouterBench** derives calibration profiles from a hash-selected 20% split and evaluates the rest; both shot regimes are separate tracks.
- **ChessBench, Habermas Machine** accept every tied best answer through their scorers; `expected` holds one representative only.

## Interactive benchmarks

MiniWoB++, ScienceWorld, Boxoban, RTFM, Hanabi and Codenames are part of the lab's 25-benchmark frozen panel but not of this static suite; the index drops them (they would otherwise be provisional zeros), and `index.json` reports the 25-panel lower bound separately.
