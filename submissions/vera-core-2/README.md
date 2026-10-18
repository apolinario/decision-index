# vera-core-2

[vera-core-2](https://huggingface.co/scar-ai/vera-core-2) (Qwen3.5-2B backbone) is a typed decision model. It reads
the state, the question and the full option menu in one forward pass (full attention without a causal mask) and
scores every option with a code readout plus an option-pointer head. It does not generate text. Apache-2.0.

| | vera-core-2 |
|---|---:|
| Decision Index 0.3, public index | **41.28** |
| raw | 56.27 |
| Knowledge & Reasoning | 21.4 |
| Language Understanding | 43.9 |
| Retrieval & Classification | 53.8 |
| Tools & Automation | 61.5 |
| Arts & Human Taste | 23.9 |
| Requests | 140,620 `ok` (140,178 scoreable), 0 unsupported, 0 errors |
| Request time, 1× MI300X, one request at a time (scoring run) | median 78 · mean 92 · p80 80 ms |
| Model revision | [`cfc7812b04b1c3e878e2e081b4ac39037d223d09`](https://huggingface.co/scar-ai/vera-core-2/tree/cfc7812b04b1c3e878e2e081b4ac39037d223d09) |
| Results | [runs/vera-core-2](https://huggingface.co/datasets/scar-ai/decision-index-results/tree/fe6e3c7bd626b174af5af7d62d16e8dd03812dda/runs/vera-core-2) |

## Reproduce

```sh
pip install "vera-s1[fast]==0.1.4"      # [fast] = flash-linear-attention
vera-serve --model scar-ai/vera-core-2 --host 127.0.0.1 --port 8220 &
python -m decision_index pipeline --engine http --option base_url=http://127.0.0.1:8220 \
    --option model=vera-core-2 --edition 0.3 --out runs/vera-core-2
```

The run used these exact settings: `vera-serve` defaults (no flags beyond host and port), bf16 backbone with the
readout, pointer and calibration in fp32. Software: vera-s1 0.1.4, flash-linear-attention 0.5.2,
transformers 5.20.1, torch 2.13.0+rocm7.1. `vera-serve` loaded the repo's `main` branch, which was `cfc7812` when the
run started (the only later commit, `33eb02b`, edits the model card).

vera-core-2 never cuts a request: the state, the question and every option are read whole, and nothing is refused.
There is no context cap below the backbone's own, so no capacity limit is declared. The weights also load with
`transformers` (`AutoModel`, `trust_remote_code=True`), and the repo has a vLLM adapter in `engines/`.

The uploaded `results.jsonl.gz` leaves out the payloads and raw outputs (as `--compact` does), so the suite text is not
redistributed. `python -m decision_index score` on it reproduces `index.json` exactly, with kit `9eb2dbe` (the run)
and with `f5a6e4f`.
