# jiwo-0.8b and jiwo-4b

[eljiwo/jiwo-0.8b](https://huggingface.co/eljiwo/jiwo-0.8b) (revision `9ec790b`) is a full fine-tune of
`Qwen/Qwen3.5-0.8B`. [eljiwo/jiwo-4b](https://huggingface.co/eljiwo/jiwo-4b) (revision `6293264`) is a full
fine-tune of `Qwen/Qwen3.5-4B`. Each model reads one prompt per request and gives a probability for every option of
every question in one forward pass, from a small readout head. It does not generate text.

| | jiwo-0.8b | jiwo-4b |
|---|---:|---:|
| Decision Index 0.2.1 | **28.00** | **44.97** |
| Raw index | 45.22 | 58.44 |
| Area skill: knowledge · language · retrieval · tools · arts | 10.3 · 32.3 · 44.5 · 38.0 · 11.2 | 26.2 · 49.1 · 51.6 · 67.6 · 28.2 |
| Requests | 150,759, all `ok` | 150,759, all `ok` |
| Median request time (1× H100, one request at a time) | 48.0 ms | 53.3 ms |
| Results | [runs/jiwo-0.8b](https://huggingface.co/datasets/eljiwo/decision-index-results/tree/d2caa6b3d8ebe0f764bf7d1603c6d512a7d2d7ca/runs/jiwo-0.8b) | [runs/jiwo-4b](https://huggingface.co/datasets/eljiwo/decision-index-results/tree/d2caa6b3d8ebe0f764bf7d1603c6d512a7d2d7ca/runs/jiwo-4b) |

The results are compact: `payload` and `raw_output` are removed, so they contain no suite text. The kit's `score`
gives the same index on them.

## Serving

```sh
pip install "jiwo[cuda] @ git+https://github.com/jiwidi/jiwo@64db63a"
hf download eljiwo/jiwo-0.8b --revision 9ec790b7bf4c61e664bbe4b5cff7da79879de989 --local-dir jiwo-0.8b
JIWO_CHECKPOINT=jiwo-0.8b JIWO_MAX_LENGTH=65536 jiwo serve    # listens on 127.0.0.1:8765
python -m decision_index run --engine http --option base_url=http://127.0.0.1:8765 --option model=jiwo-0.8b ...
```

For jiwo-4b, use `eljiwo/jiwo-4b` at revision `6293264096acadb679b605e3162d808550852623`. The in-process engine
has no HTTP layer, so it is the fastest path:

```sh
python -m decision_index run --engine jiwo.index_engine:JiwoEngine --model eljiwo/jiwo-0.8b \
    --option revision=9ec790b7bf4c61e664bbe4b5cff7da79879de989 --option max_length=65536 ...
```

The runs used `jiwo serve` from the code before its public release. On a stratified sample of 879 requests (about
20 per benchmark, 1× H100), the public jiwo code gives the same answers: all 4,512 fields for jiwo-0.8b, with
identical probabilities, and the same option on 4,509 of 4,512 fields for jiwo-4b (mean probability difference
0.0006).

## Training data

Public datasets (train splits only) and synthetic data. No suite rows were used. A word 13-gram check of the
training rows against the 0.2.1 suite found shared source text with different labels. 435 rows share at least half
of their 13-grams with suite rows of RouterBench, BRIGHT, HoVer, Amazon ESCI, ANLI or ToolRet. 2,158 rows share at
least one 13-gram (tool descriptions) with ToolRet. Without these six benchmarks, the index is 29.38 for jiwo-0.8b
and 46.96 for jiwo-4b.
