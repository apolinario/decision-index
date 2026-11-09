# Jeffan-9B-1211

A decision model built on [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B): [tattrongvu/Jeffan-9B-1211](https://huggingface.co/tattrongvu/Jeffan-9B-1211)
(bf16, 8.95B parameters, text only, Apache 2.0), read through the engine in the model repo's
[`inference/`](https://huggingface.co/tattrongvu/Jeffan-9B-1211/tree/dc62bcf1e7ddacc70e233b9557d6e8aeb975347d/inference) (Apache 2.0). Same family as Jeffan-31b-1710 (#241) and Jeffan-4B-2010 (#259).

- **Attention (please serve it this way):** the full-attention layers read the prompt in both directions (every real token of a prompt sees every other one); the linear-attention (Gated DeltaNet) layers stay causal. Stock causal inference loads the weights without an error but answers as a different model.
- **Layer loop (part of the model):** layers 4–7, 8–11 and 12–15 (three blocks) run twice in every forward pass, with a damped update (the hidden states entering a block, `x`, become `x + (block(x) - x) / 2` after each pass; fp32 update, no cache). The checkpoint's `dm_config.json` declares it (`loop`), and the engine applies it; 2 passes is the model's setting and the one this run used. The engine has an option `loop_passes` for other settings; please keep the default. Stock vLLM cannot run the loop, so there is no vLLM path.
- One prompt per question: the checkpoint's chat template with thinking off, a fixed system line, then `State:`,
  `Question:` and `Options:` with one-token codes `A`, `B`, ... (`AA`, `AB`, ... beyond 26; 255 at most), in request order.
- Nothing is generated: one forward pass per prompt (the prompts of a request are batched, shortest first, up to
  65,536 tokens per batch); the final hidden state at the last prompt token times the LM-head rows of the codes,
  softmax over the question's options. `choice`: argmax and every option's probability; `noul`: P(true); `score`: the
  expected level. One read, no re-read.

| | |
|---|---|
| Decision Index 0.3 public index | 70.55 (raw 77.58, breadth 69.79) |
| Areas | knowledge 56.8 · language 72.7 · retrieval 74.3 · tools 88.6 · arts 60.0 |
| Requests | 140,178 of 140,178 scoreable (140,620 in the edition, HLE included), all `ok`, none unsupported, no errors |
| Request time in `scores.json` | kit sequential runner, one request at a time, one B300 per shard: median 76 ms, mean 102 ms, p95 170 ms |

## Run

```sh
pip install torch "transformers==5.17.0" "flash-linear-attention==0.5.2"
pip install causal-conv1d==1.7.0 --no-build-isolation   # optional
git clone https://github.com/apolinario/decision-index && git -C decision-index checkout f8552a1 && pip install -e ./decision-index
hf download tattrongvu/Jeffan-9B-1211 --revision dc62bcf1e7ddacc70e233b9557d6e8aeb975347d --include "inference/*" --local-dir jeffan && pip install ./jeffan/inference
python -m decision_index pipeline --engine jeffan_engine:DecisionEngine \
    --option model=tattrongvu/Jeffan-9B-1211 --option revision=dc62bcf1e7ddacc70e233b9557d6e8aeb975347d \
    --option max_len=65536 --option budget=65536 \
    --out runs/Jeffan-9B-1211 --compact
```

The engine refuses any other option (except `loop_passes`, whose default is the model's 2) and any checkpoint whose
`dm_config.json` is not this model's reading (bidirectional, code readout, one read). PyTorch SDPA attention, bf16
autocast, no cache (the layer loop needs none).

## One 96 GB GPU

The weights are 17.9 GB. On one RTX PRO 6000 Blackwell Server Edition (97,887 MiB, 600 W), the kit's sequential runner with
this engine and options: 500 suite requests (the 200 largest by input tokens, up to 263,988 tokens in one request, the 100
longest states and 200 random ones) all answered, 100.00% the same answer as this run, GPU peak 44.9 GB (nvidia-smi). On a B300
with PyTorch's memory capped at 95,000 MiB, the same 500 requests were identical to this run, with at most 29.0 GiB allocated.

## Latency

On one RTX PRO 6000 (same setup): 700 requests sampled by the board's latency-v1 design, median 55 ms, mean 165 ms, p80 122 ms;
a heavier sample with all 43 benchmarks weighted equally (750 requests) gives median 67 ms, mean 456 ms, p80 664 ms. On one
B300, one process, one request at a time with the kit's runner:

| | median | mean | p80 | p95 |
|---|---:|---:|---:|---:|
| the whole public suite (this run) | 76 ms | 102 ms | 85 ms | 170 ms |
| 700 suite requests sampled by the latency-v1 design (per-benchmark counts from `methodology.json`), after 10 warm-up requests | 74 ms | 97 ms | 77 ms | 205 ms |



## How the run was executed

The 0.3 suite rebuilt with the kit at `f8552a1` (HLE included; `suite verify`: every pinned hash matches), split into
11 interleaved shards with disjoint run ids. Each shard ran as a normal `decision_index run` with the options
above on its own B300, one process, one request at a time, loading the model and the engine from the Hub at the pinned
commit (torch 2.14.1+cu130, transformers 5.17.0, flash-linear-attention
0.5.2, causal-conv1d 1.7.0). The results files were concatenated verbatim
(140,620 rows, 140,620 unique run ids) and scored with `decision_index score --edition 0.3`; the kit at `25c6ec7` (current `main`) gives the same 70.55, every benchmark identical.

## Declared limits

Prompts over 65,536 tokens, questions with more than 255 options and question types other than `choice`, `noul` and
`score` are refused (`Unsupported`); nothing is truncated. None occurs in the public suite.

## Training data

Undisclosed. No other entrant's published leaderboard checkpoint was used (no fine-tuning, merging or distillation of
other entrants' models). A training-data declaration, including our decontamination against the public suite, is
available on request.
