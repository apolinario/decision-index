# Chon-1.1-31B

[ninhnv/chon-1.1-31b](https://huggingface.co/ninhnv/chon-1.1-31b/tree/4d9e606b656e2b3d99f072b26dfa37d058dc9d85) is a LoRA adapter (rank 32) on
[google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it), loaded on the unmodified base model in bf16.
Apache-2.0. It reads each question's option letters at one answer position and returns a probability for every option;
it generates nothing and has no thinking mode. The engine is `chon.dindex:ChonEngine` from
[github.com/xoai/chon](https://github.com/xoai/chon) at tag `v1.5.3` (`860fbe4677115093233e70819354f333c01f3047`).

| | |
|---|---|
| Decision Index 0.3 public index | **65.47** (raw 73.47, breadth 64.89) |
| Requests | 140,178 of 140,178 scoreable answered; 0 unsupported, 0 errors |
| Request time in `scores.json` | median 153.8 ms, mean 725.1 ms (one request at a time per process, one process per GPU, 4 x RTX PRO 6000 Blackwell Server Edition) |
| Request time, single process | median 157 ms, mean 425 ms, p80 616 ms on 990 sampled public rows, weighted to the suite's benchmark mix, one RTX PRO 6000 Blackwell Server Edition, one request at a time |

## Run

```sh
pip install "chon[serve] @ git+https://github.com/xoai/chon@v1.5.3"
pip install -e decision-index                                   # this kit at 25c6ec7
python -m decision_index pipeline --engine chon.dindex:ChonEngine \
    --option model=ninhnv/chon-1.1-31b --compact --out runs/chon-1.1-31b
```

- **Options:** only `model`. Everything else is the model's own `chon_config.json`, stated in the run's
  `environment.json` (`model_source`).
- **Memory:** one 96 GB RTX PRO 6000 (bf16 weights, about 62 GB). Prompts over 4,096 tokens fill the cache in chunks
  of 4,096; the 512-wide full-attention heads, which PyTorch's fused attention kernels do not take, run their queries in
  chunks sized to the free GPU memory.

## What the engine does

- **Prompt:** the model's chat template with thinking off. Every question of the request is listed first (its
  instructions, in request order), then the state, then the question and its options as `(A) key: description` lines
  (yes/no as `(A) no`, `(B) yes`); the answer is the letter read after `Answer: (`. One fixed rendering for every
  benchmark: no per-benchmark prompts.
- **Readout:** a softmax over the option letters only, read with the final logit soft cap off, at per-type temperatures
  fitted on our development items (choice 0.726, yes/no 0.564, score 0.803).
- **More than 16 options:** read in groups of at most 16, then a final round between the group winners;
  P = P(group wins) x P(option | group).
- **Second read:** a choice or yes/no answer whose top probability is under 0.9, in a prompt of at most 2,000 tokens,
  is read again with the request written out twice, and the two reads are averaged.
- **Shared prefix:** the questions of one request share one read of their common prompt; each question then reads its
  own tokens against the cached keys and values (the same answers as separate passes, up to rounding).
- **Out of GPU memory:** the request is read once more with a quarter of the attention, batch and pack budgets (the same
  computation in smaller pieces); if it still does not fit it is reported unsupported. 32 requests of this run took the
  retry; all were answered.
- **Never:** it never truncates, drops options, sees gold labels or tunes anything per benchmark.

**Declared limits:** states up to 32,768 tokens (longer states are refused as unsupported, never cut); up to 256
options per question. No request of the public suite reached them (0 unsupported).

## How the submitted run was made

- The kit at `25c6ec7` (runner and scorer), the public serving code `v1.5.3` installed from GitHub in a fresh
  environment (torch 2.15.0, transformers 5.22.1), the released package (the files now in the model repo; only the
  model card was updated after the run).
- 4 x NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB), one `decision_index run --engine chon.dindex:ChonEngine
  --compact` process per GPU, one request at a time. The 140,620 rows were split into four parts by case
  (`chon.dindex shard`), each part read longest state first. When three parts had finished, the remaining rows of the
  fourth were shared among the freed GPUs (same package, code and settings, their own part folders), and every part
  folder was merged by run id with `chon.dindex merge` (a row answered twice keeps one answer). `environment.json` says
  "8 parallel shards" for that reason.
- `scores.json`: `"complete": true`, 140,178 of 140,178 scoreable requests answered.

## Training data and decontamination

208,173 typed-decision items: 63,171 from earlier Chon rounds and 145,002 new ones (public datasets turned into typed
decisions, multi-question formats, and synthetic decision tasks written and labelled by LLMs, then filtered).

- **Public datasets:** the new items include train-split items of public datasets, some of which also have Decision
  Index benchmarks (BANKING77, CLINC150, GSM8K, HellaSwag, WinoGrande and ESCI among them), rendered in our own prompt
  format. Never their test items, and no suite rows.
- **Checks:** every training item was checked against all 0.3 public suite rows (exact matches, near-duplicates and
  long shared passages). In the new items this removed 284 (233 GSM8K, 51 ESCI); the earlier rounds' items went
  through the same check.
- **Selection:** no prompt, setting or checkpoint was chosen per benchmark; the release was chosen on our own held-out
  sets and a fixed 20,000-request sample of the public suite (its comparison half), and confirmed once on the sealed
  half.
