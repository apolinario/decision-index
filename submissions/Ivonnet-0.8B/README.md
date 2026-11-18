# Ivonnet-0.8B

Complete Decision Index 0.3 public runs, **text and vision**, for [Ivonnet-0.8B](https://huggingface.co/illuin/Ivonnet-0.8B/tree/191693c76a22577657c47cd8d77569bc774946fa) (`illuin/Ivonnet-0.8B`), a typed decision model fine-tuned from Qwen/Qwen3.5-0.8B by ILLUIN Technology, by distillation from [IVE-4B](https://huggingface.co/illuin/IVE-4B) (submitted separately). Each question is answered in one forward pass with a probability for every option; nothing is generated.

| Measurement | Text | Vision |
|---|---:|---:|
| Public index | **41.27** (raw 56.66) | **55.99** |
| Requests | 140,620 run, 140,178 scored, all `ok` | 12,210 rows, all `ok` |
| `complete` | `true` (HLE included) | `true` (Winoground included) |
| Latency on the RTX PRO 6000, single process (median / mean / p80) | 38 / 49 / 40 ms | 82 / 81 / 123 ms per row |

- **Results:** [illuin/Ivonnet-0.8B-decision-index](https://huggingface.co/datasets/illuin/Ivonnet-0.8B-decision-index/tree/4872174e00def41bfc71b89ba66ea677d790c57a) (compact: `payload` and `raw_output` dropped from each row, nothing else changed; `runs/ivonnet-text/`, `runs/ivonnet-vision/`: `results.jsonl.gz`, `scores.json` or `vision-scores.json`, `index.json`, `benchmark-summary.json`, `status.json`, `environment.json`).
- **Model and code:** `illuin/Ivonnet-0.8B` at `191693c76a22577657c47cd8d77569bc774946fa`. The run was made when the repo was named `illuin/IVonnet`, at revision `4928651c35b4b409dcc10c79a1a29bbfa5eb50ba` (recorded in `environment.json`); the repo's history was then reset to a single commit before it was made public. Weights, readout, `decision_config.json` and `source/src/` are byte-identical to the evaluated revision: the sha256 of every file is in `release-manifest.json` (`model.safetensors`: `efa1c618389747c10cac685d832788fe0a147e3a0058c30643f9f7c3c7bd164c`). Later commits change only `README.md`. `source/` is the same `decision_model` package as IVE-4B's, with the kit engines and a `/v1/systemone` + `/v1/decisions` server; `source/RUNNING.md` is a self-contained guide.
- **Kit:** `apolinario/decision-index` `25c6ec7`, edition 0.3, unmodified runner (SHA256 `f4c9f36c4c557f537f1d8184db7bf00c5a84d9f4f0aa67f8c06198cead25818d`) and scorer. Both suites rebuilt with the kit; `suite import` and `vision verify` passed.
- **Runtime:** one HF Job per suite (2026-11-09, flavor `rtx-pro-6000`), torch 2.11.0+cu128, transformers 5.22.1, flash-linear-attention 0.5.2, no causal-conv1d. Text took 1 h 56, vision 17 min, one process on one GPU, so the latencies above were measured with the GPU unshared.
- **Settings (fixed, they define the model):** bf16 autocast, non-causal full attention (checked at load time), fitted temperature 1.4467 from `decision_config.json`, a request's questions batched by length under 32,768 padded tokens, images at native resolution capped at 1,638,400 px.
- **Limits:** a request over 32,768 tokens or a choice over 255 options is refused as `unsupported`, never truncated. Neither happened in these runs.
- **Retries:** none; every request was answered in one pass.

## Reproduce

```sh
hf download illuin/Ivonnet-0.8B --revision 191693c76a22577657c47cd8d77569bc774946fa --local-dir Ivonnet
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e "Ivonnet/source[fast,server]" "transformers==5.22.1"
git clone https://github.com/apolinario/decision-index && git -C decision-index checkout 25c6ec7
pip install -e "decision-index[rebuild,vision]"
cd decision-index   # after building suite-0.3/ and work/vision/ (README, or source/RUNNING.md §6.1)
python -m decision_index pipeline --engine decision_model.engine:DecisionModelEngine \
    --option model=../Ivonnet --out runs/ivonnet-text
python -m decision_index vision pipeline --dir work/vision \
    --engine decision_model.engine:DecisionModelVisionEngine --option model=../Ivonnet --out runs/ivonnet-vision
```

A `/v1/systemone` server is also available (`python -m decision_model.server --model ../Ivonnet --port 8000`, then the kit's `http` engine); the in-process engines above are the evaluated path.
