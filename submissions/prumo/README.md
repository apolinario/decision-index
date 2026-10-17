# Prumo

[tardellirs/prumo](https://huggingface.co/tardellirs/prumo) ([GitHub mirror](https://github.com/tardellirs/prumo)) is a
**rule-based decision engine with no learned parameters**. A fixed list of generic capabilities reads the *structure*
of a request and computes an answer when it can:

- code with a `def` is executed in a sandbox;
- a FEN position is searched with a node-limited alpha-beta;
- tool schemas are validated against the request;
- candidates are ranked with BM25;
- Boolean, date, ordering and arithmetic problems are solved exactly.

A question that no capability claims gets the uniform distribution, so its expected score is chance. The engine
never looks at benchmark names, item ids or answer tables, and the response format is the same for every request.

| | Prumo |
|---|---:|
| Decision Index 0.3, public index | **20.14** (raw 38.96) |
| Area skill (0.3) | Knowledge & Reasoning 17.0 · Language Understanding 3.1 · Retrieval & Classification 26.6 · Tools & Automation 45.3 · Arts & Human Taste 13.4 |
| Requests | 140,178 `ok`, 0 `unsupported`, 0 errors (`scores.json`: `"complete": true`) |
| Request time, 1 CPU thread (Apple M2), one request at a time | median 0.18 ms · mean 5.1 ms · p80 0.62 ms · p99 133 ms · max 404 ms |
| Results | [runs/prumo](https://huggingface.co/datasets/tardellirs/decision-index-results/tree/a54aa7e6569d705db24aa1bf439f49754db8dcfe/runs/prumo) (compact: no payloads, no suite text) |
| Code | revision `b445c46` |

**Size.** Prumo has **0 learned parameters**. It has no weights, no embeddings and no corpus statistics, only about
700 hand-written constants: chess piece-square tables, stop-word lists, BM25 `k1`/`b` and a few thresholds, all listed
in the card. The size chart plots `log10(size)` and only models with a positive size, so a 0 does not appear on it.
If it helps, please plot Prumo at **700** (its hand-set constants). Otherwise 0 is the honest count.

**How to run** (kit `9eb2dbe`, edition 0.3, one pass, engine defaults):

```sh
git clone https://github.com/apolinario/decision-index && pip install -e decision-index
hf download tardellirs/prumo --revision b445c46915ef5439fc4239e06f4cceee963954a4 --local-dir prumo && cd prumo
pip install -r requirements.txt          # chess==1.11.2 (move generation only)
python -m decision_index pipeline --engine decision_index_engine:DecisionEngine --edition 0.3 --out runs/prumo
```

**Determinism.** Every budget is counted in operations, not time: 7,000 chess nodes, 10⁶ executed lines of code and
60k arithmetic operations. Answers therefore do not depend on the machine. Two checks were run:

- **Fresh process, different `PYTHONHASHSEED`:** 1,100 requests, bit-identical.
- **Linux vs macOS:**
  - **Requests:** 3,370 requests, 80 from each of the 43 datasets.
  - **Linux run:** Ubuntu 24.04, CPython 3.12.3, no OS sandbox available.
  - **macOS run:** CPython 3.12.13, `sandbox-exec`.
  - **Result:** the same answers (agreement 1.0, maximum probability difference 0.0).

**Executing request code.** Code runs only in a child `python -I -S -B` inside a fresh temp directory. The
protections are:

- an OS sandbox when available: `sandbox-exec`, `bwrap` or `unshare -rn`;
- a CPU limit, a memory limit and a ban on new processes;
- a CPython audit hook that blocks network, subprocess, `ctypes`, and file reads and writes outside the temp dir and
  the Python installation;
- a deterministic line budget.

`tests/test_engine.py` checks each layer. To run without executing any request code:

```sh
--option solvers=chess,tools,retrieval,grounding,consensus,labels,music,logic,puzzles,dates,arith
```

**What was removed before this run.** An earlier prototype scored 24.81 on the public index. Some of its parsers
were keyed to the fixed wording of particular benchmarks, and a few of its rules came from reading errors on the
public test. All of these were removed:

- the CLadder template parser;
- the synthetic-phishing URL heuristics;
- the BIG-Bench Hard phrase anchors;
- the API-Bank and When2Call dialogue boosts.

The remaining thresholds were chosen on training/dev splits and on generated paraphrases. The public suite was then
run for measurement only, with no tuning after it. The card declares the few generic constants that were first tried
on public subsets before they were frozen: the chess evaluation, the chord templates, the arithmetic budget and the
consensus scoring.
