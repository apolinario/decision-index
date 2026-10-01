# Sifr 0.8B v3.1

| model | results dataset | engine | hardware |
|---|---|---|---|
| Sifr 0.8B v3.1 | mohamedlotfy50/sifr-0.8b-v3-results | sifr_engine — length-normalized key log-probabilities (acc_norm convention), fixed rendering, no truncation/filtering | 2x RTX 5060 Ti |

**Index 26.88 · raw 44.2 · coverage 100% · complete: true** (edition 0.2.1; `scores.json` in the dataset repo).

- Base: Qwen3.5-0.8B, full fine-tune (best checkpoint at epoch 1.28 of 3 scheduled, selected by held-out probes)
- Results: `runs/v31-official/scores.json` (this dataset repo), produced by the kit runner with the declared engine; untouched, re-scorable
- Trained rows (train splits only, never suite rows): BANKING77, CLINC150+OOS, ANLI, XNLI, HellaSwag, GSM8K, MMLU-aux, WinoGrande, ARC, AG News, SST-5, Emotion, FinEntity, iSarcasmEval, PhishNChips, CLadder v1, ContractNLI, RAGTruth, SATA-Bench, VAST, BPoMP, Humicroedit, Amazon ESCI, ForecastBench (resolved historical sets), chess, board-game synthetics, typed-decisions, binary-verify synthetics, HumanEval
- Engine policy declared in `environment.json`; no truncation, no option filtering, no prompt tuning
- Median latency 46.9 ms/request (RTX 5060 Ti, own measurement — inside the 1,000 ms bar)
