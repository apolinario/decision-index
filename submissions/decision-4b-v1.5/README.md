# Decision 4B v1.5 (FlyMy.AI)

[flymy-ai/decision-4b-v1.5](https://huggingface.co/flymy-ai/decision-4b-v1.5) is `Qwen/Qwen3.5-4B` (revision
`851bf6e8`) with a LoRA adapter (rank 16), folded in at load. It reads one prompt per question and returns a probability
for every option from the option letters' logits at the last prompt token. It does not generate text.

This run uses the [`decision-index` branch](https://huggingface.co/flymy-ai/decision-4b-v1.5/tree/a80e804fdc1545f671d494c411bc025adc807bff)
(revision `a80e804`). It has the same weights, prompt and temperatures as `main` (`6d173f9`, the revision JevBench
measured), with two changes for this suite:

- A choice with more than 26 options is read in two levels: one pass per block of up to 26 options, in their given
  order, then one pass over the blocks' leaders. Up to 26 options the output is identical to `main`.
- The input cap is 131,072 tokens instead of 16,384.

| | Decision 4B v1.5 |
|---|---:|
| Decision Index 0.3, public index | **38.24** (raw 53.58) |
| Area skill: knowledge · language · retrieval · tools · arts | 23.1 · 41.7 · 44.2 · 53.7 · 28.2 |
| Requests | 140,178 scoreable; 140,178 ok |
| Median request time (1× RTX 4090, one request at a time, this run) | 32.0 ms (mean 172.9 ms) |
| Results | [runs/decision-4b-v1.5-0.3](https://huggingface.co/datasets/flymy-ai/decision-index-results/tree/5ee0db3c544652d10f23385e81c4f9175f34d675/runs/decision-4b-v1.5-0.3) |

The results are compact (`payload` and `raw_output` removed), so they contain no suite text.

## How it was run

```sh
hf download flymy-ai/decision-4b-v1.5 --revision a80e804fdc1545f671d494c411bc025adc807bff --local-dir decision-4b-v1.5
hf download Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --local-dir qwen3.5-4b
pip install "transformers==5.17.0" "peft==0.21.0" "flash-linear-attention==0.5.2"
python decision-4b-v1.5/server.py --assets qwen3.5-4b --port 8090      # POST /v1/systemone, GET /health
python -m decision_index pipeline --engine http --option base_url=http://127.0.0.1:8090 \
    --option model=flymy-ai/decision-4b-v1.5 --edition 0.3 --out runs/decision-4b-v1.5-0.3 --compact
```

- Kit `9eb2dbe`; the suite rebuilt with `suite rebuild --edition 0.3` and checked by `suite verify`.
- One RTX 4090 (24 GB), bf16, one request at a time. The server records CUDA graphs for prompts up to 4,096 tokens at
  start-up (the fastest path) and runs longer prompts eagerly. The run was split across 2 machine sessions and
  resumed from its `results.jsonl`; server, weights and settings were the same throughout.
- `model.load()` checks every package file against `manifest.json` and refuses symlinks, so load from a real
  directory, not a Hugging Face cache snapshot. The flash-linear-attention kernels need a C compiler at run time.

## Declared limits

- The input cap is 131,072 tokens and nothing is truncated. A longer input, or one that does not fit on the GPU, is
  answered as unsupported. No request was over either limit.
- At most 676 options per choice. The suite has at most 255.

## Training data

Public datasets and synthetic data written and checked with open-weight models. No suite rows were used. A word
13-gram check against the 0.3 suite flags the 123 ContractNLI requests: training items built from 27 ContractNLI
train-split documents share the 17 standard hypotheses and contract boilerplate; none of those documents is a test
document. Two HoVer requests share a 13-gram with training text.
