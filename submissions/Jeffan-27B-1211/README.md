# Jeffan-27B-1211

A decision model built on [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B): [tattrongvu/Jeffan-27B-1211](https://huggingface.co/tattrongvu/Jeffan-27B-1211)
(bf16, 26.90B parameters, text only, Apache 2.0), read through the engine in the model repo's
[`inference/`](https://huggingface.co/tattrongvu/Jeffan-27B-1211/tree/cd5bd6394db2e15f12b9594f24571d6154e6afdb/inference) (Apache 2.0). Same family as Jeffan-31b-1710 (#241) and Jeffan-4B-2010 (#259).

- **Attention (please serve it this way):** the full-attention layers read the prompt in both directions (every real token of a prompt sees every other one); the linear-attention (Gated DeltaNet) layers stay causal. Stock causal inference loads the weights without an error but answers as a different model.
- **Layer loop (optional):** layers 24–35 and 36–47 (two blocks) can run more than once per forward pass, with a damped update. The checkpoint's `dm_config.json` declares it (`loop`) with **1 pass as the default** (each layer once), the setting this run used and the one to evaluate; the engine's option `loop_passes=2` runs the blocks twice (our own complete run at 2 passes: 73.52). Please keep the default.
- One prompt per question: the checkpoint's chat template with thinking off, a fixed system line, then `State:`,
  `Question:` and `Options:` with one-token codes `A`, `B`, ... (`AA`, `AB`, ... beyond 26; 255 at most), in request order.
- Nothing is generated: one forward pass per prompt (the prompts of a request are batched, shortest first, up to
  65,536 tokens per batch); the final hidden state at the last prompt token times the LM-head rows of the codes,
  softmax over the question's options. `choice`: argmax and every option's probability; `noul`: P(true); `score`: the
  expected level. One read, no re-read.

| | |
|---|---|
| Decision Index 0.3 public index | 73.39 (raw 79.65, breadth 72.84) |
| Areas | knowledge 62.5 · language 75.3 · retrieval 76.1 · tools 89.5 · arts 61.9 |
| Requests | 140,178 of 140,178 scoreable (140,620 in the edition, HLE included), all `ok`, none unsupported, no errors |
| Request time in `scores.json` | kit sequential runner, one request at a time, one B300 per shard: median 110 ms, mean 173 ms, p95 370 ms |

## Run

```sh
pip install torch "transformers==5.17.0" "flash-linear-attention==0.5.2"
pip install causal-conv1d==1.7.0 --no-build-isolation   # optional
git clone https://github.com/apolinario/decision-index && git -C decision-index checkout f8552a1 && pip install -e ./decision-index
hf download tattrongvu/Jeffan-27B-1211 --revision cd5bd6394db2e15f12b9594f24571d6154e6afdb --include "inference/*" --local-dir jeffan && pip install ./jeffan/inference
python -m decision_index pipeline --engine jeffan_engine:DecisionEngine \
    --option model=tattrongvu/Jeffan-27B-1211 --option revision=cd5bd6394db2e15f12b9594f24571d6154e6afdb \
    --option max_len=65536 --option budget=65536 \
    --out runs/Jeffan-27B-1211 --compact
```

The engine refuses any other option (except `loop_passes`, whose default is the checkpoint's) and any checkpoint whose
`dm_config.json` is not this model's reading (bidirectional, code readout, one read). PyTorch SDPA attention, bf16
autocast, no cache (the layer loop needs none).

## One 96 GB GPU

The weights are 53.8 GB. On one RTX PRO 6000 Blackwell Server Edition (97,887 MiB, 600 W), the kit's sequential runner with
this engine and options: 500 suite requests (the 200 largest by input tokens, up to 263,988 tokens in one request, the 100
longest states and 200 random ones) all answered, 99.94% the same answer as this run, GPU peak 79.5 GB (nvidia-smi; 87.5 GB on
the heavier latency sample below).

## Latency

On one RTX PRO 6000 (same setup): 700 requests sampled by the board's latency-v1 design, median 99 ms, mean 553 ms, p80 265 ms;
a heavier sample with all 43 benchmarks weighted equally (750 requests) gives median 138 ms, mean 1,332 ms, p80 1,603 ms (the
long-prompt benchmarks). On one B300, one process, one request at a time with the kit's runner:

| | median | mean | p80 | p95 |
|---|---:|---:|---:|---:|
| the whole public suite (this run) | 110 ms | 173 ms | 121 ms | 370 ms |
| 700 suite requests sampled by the latency-v1 design (per-benchmark counts from `methodology.json`), after 10 warm-up requests | 86 ms | 154 ms | 113 ms | 397 ms |



## How the run was executed

The 0.3 suite rebuilt with the kit at `f8552a1` (HLE included; `suite verify`: every pinned hash matches), split into
29 interleaved shards with disjoint run ids. Each shard ran as a normal `decision_index run` with the options
above on its own B300, one process, one request at a time, loading the model and the engine from the Hub at the pinned
commit (torch 2.14.1+cu130, transformers 5.17.0, flash-linear-attention
0.5.2, causal-conv1d 1.7.0). The results files were concatenated verbatim
(140,620 rows, 140,620 unique run ids) and scored with `decision_index score --edition 0.3`; the kit at `25c6ec7` (current `main`) gives the same 73.39, every benchmark identical.

## Declared limits

Prompts over 65,536 tokens, questions with more than 255 options and question types other than `choice`, `noul` and
`score` are refused (`Unsupported`); nothing is truncated. None occurs in the public suite.

## Training data

Undisclosed. No other entrant's published leaderboard checkpoint was used (no fine-tuning, merging or distillation of
other entrants' models). A training-data declaration, including our decontamination against the public suite, is
available on request.
