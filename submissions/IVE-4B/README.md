# IVE-4B

Complete Decision Index 0.3 public runs, **text and vision**, for [IVE-4B](https://huggingface.co/illuin/IVE-4B/tree/2b5c0a6eb0461dd3ebf8f8b31d16e44b570d5b96) (`illuin/IVE-4B`), a typed decision model fine-tuned from Qwen/Qwen3.5-4B by ILLUIN Technology. Each question is answered in one forward pass with a probability for every option; nothing is generated.

| Measurement | Text | Vision |
|---|---:|---:|
| Public index | **58.51** (raw 69.09) | **71.90** |
| Requests | 140,620 run, 140,178 scored, all `ok` | 12,210 rows, all `ok` |
| `complete` | `true` (HLE included) | `true` (Winoground included) |
| Latency on the RTX PRO 6000, single process (median / mean / p80) | 51 / 99 / 63 ms | 142 / 135 / 184 ms per row |

- **Results:** [illuin/IVE-4B-decision-index](https://huggingface.co/datasets/illuin/IVE-4B-decision-index/tree/828e32772bb1c77a3924cbab675889adccfc2a29) (compact: `payload` and `raw_output` dropped from each row, nothing else changed; `runs/ive-text/`, `runs/ive-vision/`: `results.jsonl.gz`, `scores.json` or `vision-scores.json`, `index.json`, `benchmark-summary.json`, `status.json`, `environment.json`).
- **Model and code:** `illuin/IVE-4B` at `2b5c0a6eb0461dd3ebf8f8b31d16e44b570d5b96`, the evaluated revision. Later commits change only `README.md`, `LICENSE`, `NOTICE` and `source/RUNNING.md` (the repo was also renamed from `illuin/IVE`, which redirects); weights, readout, `decision_config.json` and `source/src/` are byte-identical. `source/` is the `decision_model` package with the kit engines and a `/v1/systemone` + `/v1/decisions` server; `source/RUNNING.md` is a self-contained guide.
- **Kit:** `apolinario/decision-index` `25c6ec7`, edition 0.3, unmodified runner (SHA256 `f4c9f36c4c557f537f1d8184db7bf00c5a84d9f4f0aa67f8c06198cead25818d`) and scorer. Both suites rebuilt with the kit; `suite import` and `vision verify` passed.
- **Runtime:** one HF Job per suite (2026-11-09, flavor `rtx-pro-6000`), torch 2.11.0+cu128, transformers 5.22.1, flash-linear-attention 0.5.2, no causal-conv1d. Text took 3 h 52, vision 28 min, one process on one GPU, so the latencies above were measured with the GPU unshared.
- **Settings (fixed, they define the model):** bf16 autocast, non-causal full attention (checked at load time), fitted temperature 1.5329 from `decision_config.json`, a request's questions batched by length under 32,768 padded tokens, images at native resolution capped at 1,638,400 px.
- **Limits:** a request over 32,768 tokens or a choice over 255 options is refused as `unsupported`, never truncated. Neither happened in these runs.
- **Retries:** none; every request was answered in one pass.

## Reproduce

```sh
hf download illuin/IVE-4B --revision 2b5c0a6eb0461dd3ebf8f8b31d16e44b570d5b96 --local-dir IVE
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e "IVE/source[fast,server]" "transformers==5.22.1"
git clone https://github.com/apolinario/decision-index && git -C decision-index checkout 25c6ec7
pip install -e "decision-index[rebuild,vision]"
cd decision-index   # after building suite-0.3/ and work/vision/ (README, or source/RUNNING.md §6.1)
python -m decision_index pipeline --engine decision_model.engine:DecisionModelEngine \
    --option model=../IVE --out runs/ive-text
python -m decision_index vision pipeline --dir work/vision \
    --engine decision_model.engine:DecisionModelVisionEngine --option model=../IVE --out runs/ive-vision
```

A `/v1/systemone` server is also available (`python -m decision_model.server --model ../IVE --port 8000`, then the kit's `http` engine); the in-process engines above are the evaluated path.
