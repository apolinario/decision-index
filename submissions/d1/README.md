# d1-3B and d1-omni-600M

| model | revision | public index 0.3 | answered | median / mean request time |
|---|---|---:|---:|---|
| [d1-3B](https://huggingface.co/LiquidAI/d1-3B) | `da1fe36` | 48.99 | 100% | 12.5 / 20.5 ms |
| [d1-omni-600M](https://huggingface.co/LiquidAI/d1-omni-600M) | `414f8d6` | 14.64 | 73.5% | 8.0 / 11.5 ms |

Request times are from our runs: one AMD Instinct MI325X, eager, one request at a time.

## Engine

`d1_engine:D1` ([`code/d1_engine.py`](https://huggingface.co/datasets/LiquidAI/d1-decision-index/blob/9fcaa92417b9f80d9b2086371ecfb292e4f1d91a/code/d1_engine.py))
loads the model with its own Hub code (`trust_remote_code=True`) and answers each request with
`model.system_one(state, questions)`, the call on the model cards.

- Nothing is shortened: a request the model cannot read whole is recorded as unsupported.
- d1-omni-600M reads at most 16,384 positions and keeps a question's instructions and option texts to a token
  budget. 37,141 of 140,178 requests would be cut, so they are unsupported. Its default batching needs about 52 GB
  of GPU memory on the longest requests.
- `--option compile=true` is available (CUDA graphs, NVIDIA only); these runs did not use it.

## Run

```sh
pip install "decision-index @ git+https://github.com/apolinario/decision-index@62d2f51" "transformers==5.19.0" torch
hf download LiquidAI/d1-decision-index code/d1_engine.py --repo-type dataset --local-dir . && cp code/d1_engine.py .
python -m decision_index pipeline --edition 0.3 --engine d1_engine:D1 --compact \
    --option model=LiquidAI/d1-3B --option revision=da1fe36a861f24690f27f622dca1d8688503d113 \
    --option dtype=bfloat16 --out runs/d1-3B
python -m decision_index pipeline --edition 0.3 --engine d1_engine:D1 --compact \
    --option model=LiquidAI/d1-omni-600M --option revision=414f8d6438174f5b2133a9c21a478fc42625e308 \
    --option dtype=float16 --out runs/d1-omni-600M
```

The runs were made at earlier revisions (`90122b1` and `12accb4`). Only the model cards and their assets changed
since; the weights, configuration, code and tokenizer are the ones that ran.
