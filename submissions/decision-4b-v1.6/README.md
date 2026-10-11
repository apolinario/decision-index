# Decision 4B v1.6 (FlyMy.AI)

[flymy-ai/decision-4b-v1.6](https://huggingface.co/flymy-ai/decision-4b-v1.6) is `Qwen/Qwen3.5-4B` (revision
`851bf6e8`) with a LoRA adapter (rank 16), folded in at load. It reads one prompt per question and returns a probability
for every option from the option letters' logits at the last prompt token. It does not generate text. It supersedes
Decision 4B v1.5 (#130): a new training run on a broader decision mixture, with the same base, prompt, readout and serving.

This run uses the published package, revision [`3399e91`](https://huggingface.co/flymy-ai/decision-4b-v1.6/tree/3399e91d615950ff53e08b41fc2ede170503f9a6).
A choice with more than 26 options is read in two levels: one pass per block of up to 26 options, in their given order,
then one pass over the blocks' leaders. Inputs are capped at 131,072 tokens.

| | Decision 4B v1.6 |
|---|---:|
| Decision Index 0.3, public index | **42.69** (raw 57.41) |
| Area skill: knowledge · language · retrieval · tools · arts | 26.4 · 46.3 · 45.3 · 59.8 · 39.0 |
| Requests | 140,178 scoreable; 140,178 ok |
| Median request time (1× RTX 4090, one request at a time, this run) | 30.8 ms (mean 170.9 ms) |
| Results | [runs/decision-4b-v1.6-0.3](https://huggingface.co/datasets/flymy-ai/decision-index-results/tree/1743d914d31e2e05d3fb162a280dd3069ca60d11/runs/decision-4b-v1.6-0.3) |

The results are compact (`payload` and `raw_output` removed), so they contain no suite text.

## How it was run

```sh
hf download flymy-ai/decision-4b-v1.6 --revision 3399e91d615950ff53e08b41fc2ede170503f9a6 --local-dir decision-4b-v1.6
hf download Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --local-dir qwen3.5-4b
pip install "transformers==5.17.0" "peft==0.21.0" "flash-linear-attention==0.5.2"
python decision-4b-v1.6/server.py --assets qwen3.5-4b --port 8090      # POST /v1/systemone, GET /health
python -m decision_index pipeline --engine http --option base_url=http://127.0.0.1:8090 \
    --option model=flymy-ai/decision-4b-v1.6 --edition 0.3 --out runs/decision-4b-v1.6-0.3 --compact
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

Public datasets and synthetic data written and checked with open-weight models. No suite rows were used. Every row added
since v1.5 was checked against the 0.3 suite: rows sharing half of their word 13-grams with a suite row, or a short text
equal to one, were dropped, and most new sources also dropped any row sharing a single 13-gram. As in v1.5, the 123
ContractNLI requests share the 17 standard hypotheses and contract boilerplate with training items built from 27
ContractNLI train-split documents; none of those documents is a test document.
