# Duumvir Decider

[tardellirs/duumvir-decider](https://huggingface.co/tardellirs/duumvir-decider/tree/588eedb63a4a8b3451fcf851d5e95a29c357e40f) serves two Apache-2.0 decision models on one set of weights.
[karpo-decision-31B](https://huggingface.co/MachinePulseAI/karpo-decision-31B) (unchanged) answers every request.
[KnowLine-31B-Gen2](https://huggingface.co/PelaAI/KnowLine-31B-Gen2) runs on karpo's tensors plus a low-rank adapter
(both are Gemma 4 31B fine-tunes) and is consulted when karpo's minimum confidence is below 0.9 and the request has at
most 4 questions and 4,000 characters, or when karpo refuses a request; then probabilities are averaged 0.6 / 0.4.

| | Duumvir Decider |
|---|---:|
| Decision Index 0.3, public index | **75.42** (raw 81.19) |
| Requests | 140,620 `ok` (140,178 scored), 0 `unsupported`, 0 errors (`"complete": true`) |
| Parameters | 35,682,808,172 = 31,273,088,876 (karpo-decision-31B) + 4,409,719,296 (adapter) |
| Latency (A100 80 GB) | median 235 ms, mean 616 ms, p80 680 ms |
| Results | [runs/duumvir-decider](https://huggingface.co/datasets/tardellirs/duumvir-decider-decision-index/tree/75e52db383e49b92cf24d1a1e69b37037d65e7a6/runs/duumvir-decider) (compact; `provenance.json` explains the run) |

```sh
hf download tardellirs/duumvir-decider --revision 588eedb63a4a8b3451fcf851d5e95a29c357e40f --local-dir duumvir-decider
hf download MachinePulseAI/karpo-decision-31B --revision b8e6fd7463cea8662c556d6257955d7ef9c10d19 --local-dir karpo
hf download PelaAI/KnowLine-31B-Gen2 --revision fd99a9ed2255d8a67dd47a44cc429e638f611593 --local-dir knowline \
  --include "*.json" "*.jinja" "*.py" "tokenizer*"
cd duumvir-decider
python -m decision_index pipeline --edition 0.3 --engine duo_engine:KarpoKnowLineCascade \
  --option adapter_dir=. --option karpo_checkpoint=../karpo --option knowline_dir=../knowline --compact --out runs/duumvir-decider
```

The reported run is a complete run of this package with the kit on one A100 80 GB (spot instance, resumed from
`results.jsonl` after interruptions). Run notes are in `provenance.json`.
