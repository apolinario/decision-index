# vera-large

[vera-large](https://huggingface.co/scar-ai/vera-large) (Qwen3.8-27B backbone) is a typed decision model. It reads
the state, the question and the full option menu in one forward pass (full attention without a causal mask) and
scores every option with a code readout plus an option-pointer head. It does not generate text. Apache-2.0.

| | vera-large |
|---|---:|
| Decision Index 0.3, public index | **67.75** |
| raw | 75.55 |
| Knowledge & Reasoning | 51.2 |
| Language Understanding | 75.7 |
| Retrieval & Classification | 69.5 |
| Tools & Automation | 83.0 |
| Arts & Human Taste | 58.5 |
| Requests | 140,620 `ok` (140,178 scoreable), 0 unsupported, 0 errors |
| Request time, 1× MI300X, one request at a time (scoring run) | median 143 · mean 329 · p80 266 ms |
| Model revision | [`cc5adb16238d9ebae887da71f994cd04963d251f`](https://huggingface.co/scar-ai/vera-large/tree/cc5adb16238d9ebae887da71f994cd04963d251f) |
| Results | [runs/vera-large](https://huggingface.co/datasets/scar-ai/decision-index-results/tree/296c909d7bc85834ddc746c5e2718e9b163299ec/runs/vera-large) |

## Reproduce

```sh
pip install "vera-s1[fast]==0.1.6"      # [fast] = flash-linear-attention
vera-serve --model scar-ai/vera-large --host 127.0.0.1 --port 8240 &
python -m decision_index pipeline --engine http --option base_url=http://127.0.0.1:8240 \
    --option model=vera-large --edition 0.3 --out runs/vera-large
```

The run used `vera-serve` defaults plus `--no-caps`, which vera-large ignores (it always reads requests whole), on a
local download of the repo's `main` branch at `cc5adb1`, its current head. Software: vera-s1 0.1.6,
flash-linear-attention 0.5.2, transformers 5.20.2, torch 2.13.0+rocm7.1. The GPU was shared with another model
server during the run, so the request times above are conservative.

vera-large never cuts a request: the state, the question and every option are read whole, and nothing is refused.
There is no context cap below the backbone's own, so no capacity limit is declared. The weights also load with
`transformers` (`AutoModel`, `trust_remote_code=True`), and the repo has a vLLM adapter in `engines/`.

The uploaded `results.jsonl.gz` leaves out the payloads and raw outputs (as `--compact` does), so the suite text is not
redistributed. `python -m decision_index score` on it reproduces `index.json` exactly, with kit `9eb2dbe` (the run)
and with `25c6ec7`.
