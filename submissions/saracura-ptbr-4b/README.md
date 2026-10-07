# Saracura PT-BR 4B

[felhen-ai/saracura-ptbr-4b](https://huggingface.co/felhen-ai/saracura-ptbr-4b) (Apache-2.0) is a Kev-recipe decision
model for Brazilian Portuguese: a LoRA adapter and pointer head on `Qwen/Qwen3.5-4B-Base`, initialized from
`jaredpalmer/kev-4b` at revision `6cfce5c` (round 10 weights, listed on the 0.3 board as Kev 4B r10, 37.95) and
fine-tuned on Portuguese typed decisions with Kev's published trainer. The Portuguese fine-tune leaves the public
index where its starting checkpoint was. It is served by Kev's System One-compatible server and does not
generate text.

| | |
|---|---|
| Decision Index 0.3, public index | **37.96** |
| Raw public index | 53.80 |
| Area skill | knowledge 25.3 · language 41.9 · retrieval 46.2 · tools 53.9 · arts 14.9 |
| Requests | 153,397, all `ok`, none unsupported, no errors |
| Results | [felhen-ai/decision-index-results](https://huggingface.co/datasets/felhen-ai/decision-index-results/tree/a7eebd8bc26e229b24973c3021989ff7e6f125fc/runs/saracura-ptbr-4b) (compact, no suite text) |

The complete 0.2.1 run (37.91, runner `87d4650`) was resumed under 0.3, which ran only the 2,638 rebuilt GSM8K
requests. Its 0.2.1 scores are kept in `runs/saracura-ptbr-4b/edition-0.2.1/`.

## Running it

```sh
git clone https://github.com/jaredpalmer/kev && cd kev && uv sync --extra serve
uv run --extra serve python -m kev.serve --run felhen-ai/saracura-ptbr-4b --port 8019

python -m decision_index pipeline --engine http \
    --option base_url=http://127.0.0.1:8019 --option model=kev-latest --out runs/saracura-ptbr-4b
```

The run was stored full and compacted afterwards (`payload` and `raw_output` dropped, as `--compact` does);
re-scoring the compact `results.jsonl.gz` gives the same 37.96. Typical latency on one RTX 5090 in bf16 is about
100 to 170 ms per request for long states.

## Training data

Fine-tuning data is Portuguese only: the train splits of the PT-BR typed-decisions benchmark
([felhen-ai/ptbr-typed-decisions-bench](https://huggingface.co/datasets/felhen-ai/ptbr-typed-decisions-bench):
OLID-BR, FACTCK.BR, FaQuAD-NLI, SciELO, JurisTCU, Câmara dos Deputados), the `pt` config of
`telepatia-ai/typed-decisions-pt-es`, internal Brazilian documents and marketplace listings, and synthetic Portuguese
cases labeled by open-weight Qwen models. None of these sources is a suite benchmark. We did not audit what
`telepatia-ai/typed-decisions-pt-es` was built from, and the starting checkpoint `jaredpalmer/kev-4b` carries
whatever its own card lists.
