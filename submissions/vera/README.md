# vera-core and vera-spark

[vera-core](https://huggingface.co/scar-ai/vera-core) (Qwen3.5-2B backbone) and
[vera-spark](https://huggingface.co/scar-ai/vera-spark) (ModernBERT-large, 400M) are typed decision models: each reads
the state, the question and the full option menu in one forward pass and scores every option with a trained head.
They do not generate text. Both are Apache-2.0.

| | vera-core | vera-spark |
|---|---:|---:|
| Decision Index 0.3, public index | **30.11** | **19.59** |
| Knowledge & Reasoning | 15.4 | 10.1 |
| Language Understanding | 34.9 | 24.9 |
| Retrieval & Classification | 39.5 | 30.1 |
| Tools & Automation | 41.4 | 18.1 |
| Arts & Human Taste | 16.3 | 12.4 |
| Requests | 140,620 `ok`, 0 unsupported, 0 errors | 140,620 `ok`, 0 unsupported, 0 errors |
| Request time, 1× MI300X, one request at a time (scoring run) | median 116 · mean 163 · p80 175 ms | median 72 · mean 78 · p80 76 ms |
| Model revision | `22bf59c` | `964a12f` |
| Results | [runs/vera-core](https://huggingface.co/datasets/scar-ai/decision-index-results/tree/1929fc9f1ba7f60a31bc48d340debcfd97dfffa3/runs/vera-core) | [runs/vera-spark](https://huggingface.co/datasets/scar-ai/decision-index-results/tree/1929fc9f1ba7f60a31bc48d340debcfd97dfffa3/runs/vera-spark) |

## Reproduce

```sh
pip install "vera-s1[fast]==0.1.3"      # [fast] = flash-linear-attention, used by vera-core
vera-serve --model scar-ai/vera-core --no-caps --host 127.0.0.1 --port 8220 &
python -m decision_index pipeline --engine http --option base_url=http://127.0.0.1:8220 \
    --option model=vera-core --edition 0.3 --out runs/vera-core
```

(and the same with `scar-ai/vera-spark` / `model=vera-spark`). `--no-caps` reads every request whole: without it the
server fits requests to the trained budget by shortening long option descriptions, long questions and oversized
states, which this board does not allow. With it nothing is cut and nothing is refused (up to 255 options per
question). Inputs longer than training (8k tokens) are out of distribution but answered.

The uploaded `results.jsonl.gz` keeps only the fields the scorer reads (as `--compact` does), so the suite text is not
redistributed; `python -m decision_index score` on it reproduces `index.json` exactly.

## Training data

Train splits of public datasets (multi-hop and table QA, conditional and legal QA, reading comprehension, math and
code-judging sets, preference data), code-generated decision tasks, and synthetic decisions labelled by an open-weight
teacher. No evaluation split was used on purpose. We decontaminated against JevBench, but did not run a content
firewall against the 0.3 suite.
