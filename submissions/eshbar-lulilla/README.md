# 𒂠𒁇 𒇽𒆤𒇲 (eshbar-lulilla)

Decision Index 0.3 public score: **70.27** (self-run; raw 77.52, breadth 69.72). 140,620 requests run, 140,178 scored, 0 unsupported,
0 errors.

- Model: https://huggingface.co/plantwaterer/eshbar-lulilla/tree/c3bf893bc24ac2b4ecc89969e3ab27640ec5f5e8
- Results: https://huggingface.co/datasets/plantwaterer/eshbar-lulilla-decision-index/tree/cf4bc376937e4425aa4ffd10eaceac4b52caa6cf/runs/eshbar-lulilla
- Kit: apolinario/decision-index @ 9eb2dbe2a358004c8782c66e40a83ac07b953fec.
- Engine: `eshbar_lulilla:Engine`, in-process, from the model repository.

## Settings

```bash
REPO=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('plantwaterer/eshbar-lulilla', revision='c3bf893bc24ac2b4ecc89969e3ab27640ec5f5e8'))")
PYTHONPATH="$REPO" python -m decision_index run --engine eshbar_lulilla:Engine --option model="$REPO" --compact --out runs/eshbar-lulilla
```

- bf16, one GPU with at least 75 GB (fits one RTX PRO 6000). Inference details are in the model card.
- Latency (one RTX PRO 6000 Blackwell, 1,000-row board sample): median 296 ms, mean 810 ms, p80 932 ms.
- Third-party components: the executor derives from LOOM (Apache-2.0); the knowledge-store table is CC BY-SA 4.0. Notices are in the
  model repository.
