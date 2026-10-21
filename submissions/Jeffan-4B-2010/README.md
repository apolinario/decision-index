# Jeffan-4B-2010

A decision model fine-tuned from [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B):
[tattrongvu/Jeffan-4B-2010](https://huggingface.co/tattrongvu/Jeffan-4B-2010) (bf16, 4.21B parameters, text only,
CC BY-NC 4.0), read through the engine in the model repo's
[`inference/`](https://huggingface.co/tattrongvu/Jeffan-4B-2010/tree/fccb1ed533de290132130f1c1de916699a08efa1/inference) (MIT). Same family as
Jeffan-31b-1710 (#241).

- **Attention:** the full-attention layers read the prompt in both directions (every real token of a prompt sees every
  other one); the linear-attention (Gated DeltaNet) layers stay causal. The model was trained this way, and it has to be
  served with this mask: stock causal inference loads it without an error but gives different answers (with stock
  vLLM, 32 of 300 held-out questions chose another option). `inference/` also has a vLLM plugin (`jeffan_vllm`); it
  needs a B200 and refuses other GPUs, so on an RTX PRO 6000 please use the engine below.
- One prompt per question: the checkpoint's chat template with thinking off, a fixed system line, then `State:`,
  `Question:` and `Options:` with one-token codes `A`, `B`, ... (`AA`, `AB`, ... beyond 26; 255 at most), in request order.
- Nothing is generated: one forward pass per prompt (the prompts of a request are batched, shortest first, up to
  65,536 tokens per batch); the final hidden state at the last prompt token times the LM-head rows of the codes,
  softmax over the question's options. `choice`: argmax and every option's probability; `noul`: P(true); `score`: the
  expected level. One read, no re-read.

| | |
|---|---|
| Decision Index 0.3 public index | 64.85 (raw 73.92, breadth 63.76) |
| Areas | knowledge 49.5 · language 69.8 · retrieval 62.8 · tools 87.4 · arts 54.6 |
| Requests | 140,178 of 140,178 scoreable (140,620 in the edition, HLE included), all `ok`, none unsupported, no errors |
| Request time in `scores.json` | kit sequential runner, one request at a time, one B200 per shard: median 37 ms, mean 53 ms, p95 95 ms |

## Run

```sh
pip install torch "transformers==5.17.0" "flash-linear-attention==0.5.2"
pip install causal-conv1d==1.7.0 --no-build-isolation   # optional, see below
git clone https://github.com/apolinario/decision-index && git -C decision-index checkout f8552a1 && pip install -e ./decision-index
hf download tattrongvu/Jeffan-4B-2010 --revision fccb1ed533de290132130f1c1de916699a08efa1 --include "inference/*" --local-dir jeffan && pip install ./jeffan/inference
python -m decision_index pipeline --engine jeffan_engine:DecisionEngine \
    --option model=tattrongvu/Jeffan-4B-2010 --option revision=6ff63018c753a62df25de376e77e161d40c0e927 \
    --option max_len=65536 --option budget=65536 \
    --out runs/Jeffan-4B-2010 --compact
```

The run loaded the weights at `6ff63018`; `fccb1ed5` adds the engine to `inference/` and updates the
model card, so `revision=fccb1ed533de290132130f1c1de916699a08efa1` serves the same weights and configs. The engine refuses any other option and any checkpoint
whose `dm_config.json` is not this model's reading (bidirectional, code readout, one read). PyTorch SDPA attention,
bf16 autocast.

transformers uses flash-linear-attention and causal-conv1d for the Gated DeltaNet layers when they are installed, and
its own PyTorch code otherwise. The submitted run had both. Without causal-conv1d (it has to be built for the local
GPU), the answers agree with this run on 99.85% of questions (top-1; 6,145 questions in 1,210 suite requests) at about
the same speed. Without flash-linear-attention as well, agreement is 99.87%, but requests take about 1.7-2.7x as long
(B200), so please install flash-linear-attention.

## One 96 GB GPU

The weights are 8.4 GB. On a B200 with PyTorch's memory capped at 95,000 MiB (an RTX PRO 6000 has 97,887 MiB): the 200
suite requests with the longest prompts (31,695 tokens at most) and 500 others (long and random) all answered,
identical to this run, with at most 16.8 GiB allocated.

## Latency

Not measured on an RTX PRO 6000. On one B200, one process, one request at a time with the kit's runner:

| | median | mean | p80 | p95 |
|---|---:|---:|---:|---:|
| the whole public suite (this run) | 37 ms | 53 ms | 41 ms | 95 ms |
| `decision_index suite sample --n 750` (default seed), after 10 warm-up requests | 37 ms | 98 ms | 129 ms | 325 ms |

## How the run was executed

The 0.3 suite rebuilt with the kit at `f8552a1` (HLE included; `suite verify`: every pinned hash matches), split into
eight interleaved shards with disjoint run ids. Each shard ran as a normal `decision_index run` with the options above
on its own B200, one process, one request at a time, loading the model from the Hub at the pinned commit (torch
2.14.1+cu130, transformers 5.17.0, flash-linear-attention 0.5.2, causal-conv1d 1.7.0). The eight results files were
concatenated verbatim (140,620 rows, 140,620 unique run ids) and scored with `decision_index score --edition 0.3`; the
kit at `25c6ec7` gives the same 64.85 with every benchmark identical.

## Declared limits

Prompts over 65,536 tokens, questions with more than 255 options and question types other than `choice`, `noul` and
`score` are refused (`Unsupported`); nothing is truncated. None occurs in the public suite (its longest prompt for this
tokenizer is 31,695 tokens).

## Training data

Undisclosed. No published leaderboard checkpoint was used (no fine-tuning, merging or distillation of other entrants).
A training-data declaration, including our decontamination against the public suite, is available on request.
