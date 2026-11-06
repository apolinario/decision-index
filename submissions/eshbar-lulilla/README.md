# 𒂠𒁇 𒇽𒆤𒇲 (eshbar-lulilla)

Decision Index 0.3 public score: **70.27** (self-run; raw 77.52, breadth 69.72). 140,620 requests run, 140,178 scored, 0 unsupported,
0 errors.

- Model: https://huggingface.co/plantwaterer/eshbar-lulilla/tree/c85a0a04c74ee761bcd8a565caf2a7d1f70ac0fb
- Results: https://huggingface.co/datasets/plantwaterer/eshbar-lulilla-decision-index/tree/cf4bc376937e4425aa4ffd10eaceac4b52caa6cf/runs/eshbar-lulilla
- Kit: apolinario/decision-index @ 9eb2dbe2a358004c8782c66e40a83ac07b953fec.
- The public score above was produced with the earlier revision `c3bf893b`, whose answers this revision reproduces bit-comparably: the only change is inference speed (exact speculative decoding on the program-writing path); 0 / 2,000 certified answer flips, and the RTX certification re-run on this revision passed the latency gate on both samples.
- Engine: `eshbar_lulilla:Engine`, in-process, from the model repository.

## Settings

```bash
REPO=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('plantwaterer/eshbar-lulilla', revision='c85a0a04c74ee761bcd8a565caf2a7d1f70ac0fb'))")
PYTHONPATH="$REPO" python -m decision_index run --engine eshbar_lulilla:Engine --option model="$REPO" --compact --out runs/eshbar-lulilla
```

- bf16, one GPU with at least 75 GB (fits one RTX PRO 6000). Inference details are in the model card.
- Latency (one RTX PRO 6000 Blackwell, one request at a time; 1,000-row public board sample): median 299 ms, mean 689 ms, p80 924 ms; hard 1,000-request sample at an 8% routing rate: median 193 ms, mean 424 ms, p80 344 ms.
- Third-party components: the executor derives from LOOM (Apache-2.0); the knowledge-store table is CC BY-SA 4.0. Notices are in the
  model repository.
