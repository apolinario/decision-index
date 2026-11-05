# haimation

Decision Index 0.3 public score: **67.44** (self-run; raw 75.38, breadth 66.82). 140,620 requests run, 140,178 scored, 0 unsupported,
0 errors.

- Model: https://huggingface.co/plantwaterer/haimation/tree/25d29eded40bcf59c193bb9d42b6f680a48dd706
- Results: https://huggingface.co/datasets/plantwaterer/haimation-decision-index/tree/da51884dc075bd5891bcc66fcfe73a7b18950652/runs/haimation
- Kit: apolinario/decision-index @ 9eb2dbe2a358004c8782c66e40a83ac07b953fec.
- Engine: `haimation:Engine`, in-process, from the model repository.

## Settings

```bash
REPO=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('plantwaterer/haimation', revision='25d29eded40bcf59c193bb9d42b6f680a48dd706'))")
PYTHONPATH="$REPO" python -m decision_index run --engine haimation:Engine --option model="$REPO" --compact --out runs/haimation
```

- bf16, one GPU with at least 75 GB (fits one RTX PRO 6000). Inference details are in the model card.
- Latency (one RTX PRO 6000 Blackwell, 1,000-row board sample): median 312 ms, mean 827 ms, p80 964 ms.
- Third-party components: the executor derives from LOOM (Apache-2.0); the knowledge-store table is CC BY-SA 4.0. Notices are in the
  model repository.
