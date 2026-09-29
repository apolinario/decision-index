# Sieve-9B evaluation harness

Sieve-9B (`sthanika-ai/Sieve-9B` at `dc2208c47c34afbfc78c393d6e738634b66b4e27`) is a LoRA adapter plus a pointer head on `Qwen/Qwen3.5-9B` (`c202236`).
It reads the state once and scores every supplied option directly; it generates nothing. The engine
is `sieve.decision_index_engine:SieveEngine` from [github.com/sthanika-ai/sieve](https://github.com/sthanika-ai/sieve)
at `28aed61bb1f29919f883823acf708b45e1cdb10f`.

## Run

```sh
pip install "sieve-decisions[cuda] @ git+https://github.com/sthanika-ai/sieve@28aed61bb1f29919f883823acf708b45e1cdb10f"
hf download sthanika-ai/Sieve-9B --revision dc2208c47c34afbfc78c393d6e738634b66b4e27 --local-dir Sieve-9B
python -m decision_index run --engine sieve.decision_index_engine:SieveEngine --option model=Sieve-9B --out runs/Sieve-9B
python -m decision_index score --results runs/Sieve-9B/results.jsonl
```

The backbone is downloaded at its pinned revision on first use, and about 20 GB of GPU memory is needed.

## What the engine does

- **Serving:** the model runs exactly as it is served. The adapter is merged into the bf16 backbone, the pointer
  head runs in fp32, and probabilities are divided by the temperature stored in `head.pt` (1.631), which changes no
  answer.
- **One question:** the state and the question run as one causal row.
- **Several questions:** the state is prefilled once and the questions continue from its cache, in batches sized to
  a memory budget. Batching changes no answer.
- **What it never does:** truncate, drop options, tune prompts per benchmark or see gold labels. `noul` criteria
  descriptions are kept as the yes/no option texts.
- **Declared limits:** a 65,536-token state, a 32,768-token question and up to 255 choice options. Requests beyond them
  are `Unsupported`; no 0.2.1 request reached them.
- **Out of memory:** raised as an ordinary error, so a resumed run retries it. None occurred.
- **Latency:** the engine times the eager path. The package's server (`sieve-serve --graphs`) also has a CUDA-graph
  path, which is faster at small option counts.
