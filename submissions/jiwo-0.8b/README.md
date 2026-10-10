# jiwo-0.8b v2

This run updates the existing **jiwo 0.8B** entry. [eljiwo/jiwo-0.8b](https://huggingface.co/eljiwo/jiwo-0.8b) v2
is at revision `71b7911` (Hub tag `v2`). The board entry today is v1 (Hub tag `v1`, `a3a37e4`, the same weights with a
later card). Both are full fine-tunes of `Qwen/Qwen3.5-0.8B` with the same recipe. v2 has a larger training mix that
adds many new domains. The model reads one prompt per request and gives a probability for every option of every question in one
forward pass, from a small readout head. It does not generate text.

| | jiwo-0.8b v2 | jiwo-0.8b v1 (board) |
|---|---:|---:|
| Decision Index 0.3, public index | **29.58** (raw 47.20) | 28.72 |
| Decision Index 0.2.1 | 28.72 (raw 45.95) | 28.00 |
| Area skill (0.3): knowledge · language · retrieval · tools · arts | 10.9 · 36.1 · 44.2 · 40.3 · 12.2 | |
| Requests | 0.3: 140,178 scoreable, all `ok`, 0 unsupported | |
| Median request time (1× H100, one request at a time, submitted run) | 47.3 ms | 48 ms |
| Median request time (1× RTX PRO 6000, 879-request sample, `JIWO_CUDA_GRAPHS=1`) | **13.1 ms** (26.6 ms without graphs) | |
| Results | [runs/jiwo-0.8b-v2-0.3](https://huggingface.co/datasets/eljiwo/decision-index-results/tree/960a30940884014b3b299167e73e257694416d61/runs/jiwo-0.8b-v2-0.3) and [runs/jiwo-0.8b-v2](https://huggingface.co/datasets/eljiwo/decision-index-results/tree/960a30940884014b3b299167e73e257694416d61/runs/jiwo-0.8b-v2) (0.2.1) | |

The 0.3 run is the complete 0.2.1 run resumed under 0.3 with kit `62d2f51`, so only the 2,638 rebuilt GSM8K
requests ran again, with the same server, weights and settings. Its `results.jsonl.gz` therefore also holds the 0.2.1
GSM8K rows, which 0.3 does not read. The results are compact: `payload` and `raw_output` are removed, so they
contain no suite text. The kit's `score` gives the same index on both runs (29.58 and 28.72). The runs served the
model under the name `jiwo-0.8b-v5`, the internal name of v2.

## Serving

```sh
pip install "jiwo[cuda] @ git+https://github.com/jiwidi/jiwo@752d8c3"
hf download eljiwo/jiwo-0.8b --revision 71b7911cf68f7df52fdc2dbf07de1e158bb807f2 --local-dir jiwo-0.8b
JIWO_CHECKPOINT=jiwo-0.8b JIWO_MAX_LENGTH=65536 JIWO_CUDA_GRAPHS=1 jiwo serve    # listens on 127.0.0.1:8765
python -m decision_index run --engine http --option base_url=http://127.0.0.1:8765 --option model=jiwo-0.8b ...
```

`JIWO_CUDA_GRAPHS=1` is the fastest path of jiwo `752d8c3`. At start-up the server captures CUDA graphs for fixed
batch shapes, which takes about 2 minutes. After that it replays them, and a batch outside the captured shapes runs
eagerly. Without the setting, the server runs eagerly. A model of this size spends most of an eager request on kernel
launches, so the graphs make it about 2 times faster on this sample.

The submitted runs used the training checkout of jiwo, which has the same model code. On a stratified sample of 879
requests (about 20 per benchmark) with the published v2 weights, these are the shares of fields that pick the same
option as the submitted run:

| Code and mode | GPU | Same option |
|---|---|---:|
| `752d8c3`, eager | H100 | 100.00% |
| `752d8c3`, eager | RTX PRO 6000 | 99.56% |
| `752d8c3`, `JIWO_CUDA_GRAPHS=1` | RTX PRO 6000 | 99.49% |

The differences are kernel round-off on near ties. On the RTX PRO 6000, the graph and eager servers pick the same
option for 99.71% of the fields.

## Training data

Public datasets (train splits only) and synthetic data. No suite rows were used. The v1 note about shared source text
still applies: the v1 training rows share source text with different labels with six index benchmarks (RouterBench,
BRIGHT, HoVer, Amazon ESCI, ANLI and ToolRet). The rows that v2 adds share no word 13-gram with the 0.2.1 or the 0.3 suite.
