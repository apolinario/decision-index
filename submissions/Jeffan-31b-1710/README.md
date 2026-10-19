# Jeffan-31b-1710

A decision model fine-tuned from [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it):
[tattrongvu/Jeffan-31b-1710](https://huggingface.co/tattrongvu/Jeffan-31b-1710) (bf16, CC BY-NC 4.0), read through the
engine in the model repo's [`inference/`](https://huggingface.co/tattrongvu/Jeffan-31b-1710/tree/e3900339a9bd47b354ecb918c0ce2a6d47a190b3/inference) (MIT).

- One prompt per question: the checkpoint's chat template with thinking off, a fixed system line, then `State:`,
  `Question:` and `Options:` with one-token codes `A`, `B`, ... (`AA`, `AB`, ... beyond 26; 255 at most), in request order.
- Nothing is generated: one vLLM call per request with `max_tokens=1`, `allowed_token_ids` = the codes and `logprobs` =
  the number of options under `logprobs_mode=processed_logprobs`, i.e. the next-token distribution renormalised over the
  codes. `choice`: argmax and every option's probability; `noul`: P(true); `score`: the expected level.
- A question whose top-1 probability is below 0.9 is read a second time with the input shown twice, and that
  distribution replaces the first (`reread_below=0.9`). The second prompt begins with the first one's tokens, so the
  prefix cache serves the first copy.

| | |
|---|---|
| Decision Index 0.3 public index | 73.13 (raw 79.74, breadth 72.64) |
| Areas | knowledge 63.2 · language 76.0 · retrieval 72.2 · tools 89.4 · arts 63.6 |
| Requests | 140,178 of 140,178 scoreable (140,620 in the edition, HLE included), all `ok`, none unsupported, no errors |
| Request time in `scores.json` | kit sequential runner, one request at a time, one B200 per shard: median 43 ms, mean 104 ms, p95 417 ms |

## Run

```sh
pip install "vllm==0.29.0" "transformers==5.20.0"
git clone https://github.com/apolinario/decision-index && git -C decision-index checkout f8552a1 && pip install -e ./decision-index
hf download tattrongvu/Jeffan-31b-1710 --revision e3900339a9bd47b354ecb918c0ce2a6d47a190b3 --include "inference/*" --local-dir jeffan && pip install -e ./jeffan/inference
python -m decision_index pipeline --engine dm_decision_engine:DecisionEngine \
    --option model=tattrongvu/Jeffan-31b-1710 --option revision=e3900339a9bd47b354ecb918c0ce2a6d47a190b3 \
    --option max_len=98304 --option max_batched_tokens=4096 --option reread_below=0.9 \
    --out runs/Jeffan-31b-1710 --compact
```

The engine refuses any other option. Defaults: `gpu_mem=0.85`, `max_num_seqs=256`, prefix caching on, image and video
inputs off.

## One 96 GB GPU

vLLM 0.29 reserves, for each of Gemma-4-31B's 50 sliding-window layers, (window 1,024 + `max_num_batched_tokens`)
tokens of attention cache. With vLLM's default chunk (16,384 for the offline `LLM` class) the engine does not start on
a 96 GB card at `gpu_memory_utilization=0.85`; with `max_batched_tokens=4096` (the engine's default, used in this run)
it does. Checked on a B200 limited to a 96 GB card's share (`gpu_memory_utilization=0.4538`): 126,759 tokens of
attention cache, the 150 longest and 350 random suite requests all answered, 99.91% top-1 agreement with an unlimited
run.

## One RTX PRO 6000

The same package, settings and model commit on one NVIDIA RTX PRO 6000 Blackwell Server Edition (600 W, driver
580.126.20), one process, one request at a time, with the kit's runner:

| | |
|---|---|
| start-up | KV cache 19.0 GiB = 128,376 tokens on a cold start (20.98 GiB = 141,808 with the compile cache warm); vLLM picks TRITON_ATTN on this GPU |
| the 150 longest suite requests + 350 random | 500 / 500 `ok`, peak 80.0 of 95.6 GiB in use; top-1 agreement with the submitted run 99.84% of questions |
| `decision_index suite sample --n 750` (default seed) after 10 warm-up requests | 760 / 760 `ok`; median 179 ms, mean 616 ms, **p80 1,082 ms**, p95 2,133 ms (a repeat: p80 1,078 ms); top-1 agreement 99.96% |

That sample weights every benchmark equally, and its slowest fifth is mostly requests with many questions (for
example ACOS, about 64 questions per request), each read with its own prompt. Over the whole public suite the request
times are much lower (one B200: median 43 / mean 104 / p80 101 ms; this card was about 3.1-3.4x slower than a B200 on
the sample). A lower re-read threshold barely moves that sample's p80 (it stays around 830 ms even with no re-read at
all), so we submit the setting above.
