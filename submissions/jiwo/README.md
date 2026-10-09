# jiwo-4b v2

This run updates the existing **jiwo 4B** entry. [eljiwo/jiwo-4b](https://huggingface.co/eljiwo/jiwo-4b) v2 is at
revision `9adc559` (Hub tag `v2`). The board entry today is v1, revision `6293264`. The Hub tag `v1` (`2e31a9c`) has the same weights with a later card. Both are full
fine-tunes of `Qwen/Qwen3.5-4B` with the same recipe. v2 has a larger and more varied training mix. The model reads
one prompt per request and gives a probability for every option of every question in one forward pass, from a small
readout head. It does not generate text.

| | jiwo-4b v2 | jiwo-4b v1 (board) |
|---|---:|---:|
| Decision Index 0.3, public index | **46.06** (raw 59.63) | 45.76 |
| Decision Index 0.2.1 | 45.08 (raw 58.41) | 44.97 |
| Area skill (0.3): knowledge · language · retrieval · tools · arts | 29.1 · 52.9 · 51.7 · 63.1 · 29.8 | |
| Requests | 0.3: 140,178 scoreable, all `ok`, 0 unsupported | |
| Median request time (1× H100, one request at a time, submitted run) | 48.6 ms | 53.3 ms |
| Median request time (1× RTX PRO 6000, 879-request sample, `JIWO_CUDA_GRAPHS=1`) | 31.6 ms (34.5 ms without graphs) | |
| Results | [runs/jiwo-4b-v2-0.3](https://huggingface.co/datasets/eljiwo/decision-index-results/tree/cdb86582a7117debc17da64a6e75032508c7e7ef/runs/jiwo-4b-v2-0.3) and [runs/jiwo-4b-v2](https://huggingface.co/datasets/eljiwo/decision-index-results/tree/cdb86582a7117debc17da64a6e75032508c7e7ef/runs/jiwo-4b-v2) (0.2.1) | |

The 0.3 run is the complete 0.2.1 run resumed under 0.3 with kit `62d2f51`, so only the 2,638 rebuilt GSM8K
requests ran again, with the same server, weights and settings. Its `results.jsonl.gz` therefore also holds the 0.2.1
GSM8K rows, which 0.3 does not read. The results are compact: `payload` and `raw_output` are removed, so they
contain no suite text. The kit's `score` gives the same index on both runs (46.06 and 45.08).

## Serving

```sh
pip install "jiwo[cuda] @ git+https://github.com/jiwidi/jiwo@752d8c3"
hf download eljiwo/jiwo-4b --revision 9adc559ff7ec5baffade3998f7bca85da0d124ed --local-dir jiwo-4b
JIWO_CHECKPOINT=jiwo-4b JIWO_MAX_LENGTH=65536 JIWO_CUDA_GRAPHS=1 jiwo serve    # listens on 127.0.0.1:8765
python -m decision_index run --engine http --option base_url=http://127.0.0.1:8765 --option model=jiwo-4b ...
```

The in-process engine has no HTTP layer:

```sh
python -m decision_index run --engine jiwo.index_engine:JiwoEngine --model eljiwo/jiwo-4b \
    --option revision=9adc559ff7ec5baffade3998f7bca85da0d124ed --option max_length=65536 ...
```

`JIWO_CUDA_GRAPHS=1` is the fastest path of jiwo `752d8c3`. At start-up the server captures CUDA graphs for fixed
batch shapes, which takes about 4 minutes. After that it replays them, and a batch outside the captured shapes runs
eagerly. Without the setting, the server runs eagerly.

The submitted run used jiwo `40d43c4` without CUDA graphs on an H100. On a stratified sample of 879 requests (about 20
per benchmark) with the published v2 weights, these are the shares of fields that pick the same option as the
submitted run:

| Code and mode | GPU | Same option |
|---|---|---:|
| `40d43c4`, eager (the submitted setup) | H100 | 100% (identical answers, two runs) |
| `752d8c3`, eager | H100 | 99.93% |
| `752d8c3`, eager | RTX PRO 6000 | 99.67% |
| `752d8c3`, `JIWO_CUDA_GRAPHS=1` | RTX PRO 6000 | 99.60% |

The differences are kernel round-off on near ties. On the RTX PRO 6000, the graph and eager servers pick the same
option for 99.84% of the fields.

## Training data

Public datasets (train splits only) and synthetic data. No suite rows were used. The v1 note about shared source text
still applies: the v1 training rows share source text with different labels with six index benchmarks (RouterBench,
BRIGHT, HoVer, Amazon ESCI, ANLI and ToolRet). The rows that v2 adds share no word 13-gram with the 0.2.1 suite.
