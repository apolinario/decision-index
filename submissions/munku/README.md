# 𒁵𒄩 (munku)

Decision Index 0.3 public score: **71.10** (self-run; raw 78.06, breadth 70.54). 140,620 requests run, 140,178 scored, 0 unsupported,
0 errors.

- Model: https://huggingface.co/plantwaterer/munku/tree/6b9d66a8bfb055836132e3f3f68a29e4d9dd9e95
- Results: https://huggingface.co/datasets/plantwaterer/munku-decision-index/tree/3a323f98a0a06193f11628e58527c4e22b7fbff5/runs/munku
- Kit: apolinario/decision-index @ 9eb2dbe2a358004c8782c66e40a83ac07b953fec.
- Engine: `munku:Engine`, in-process, from the model repository, with `--option route=off`. The public run was made with the engine's
  default (route on); the route changed no answer in it (0 of 3,490 routed questions produced a usable program, and the store answered none), so `route=off` gives the same answers.

## Settings

```bash
REPO=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('plantwaterer/munku', revision='6b9d66a8bfb055836132e3f3f68a29e4d9dd9e95'))")
PYTHONPATH="$REPO" python -m decision_index run --engine munku:Engine --option model="$REPO" --option route=off --compact --out runs/munku
```

- bf16, one GPU with at least 75 GB (fits one RTX PRO 6000). Inference details are in the model card.
- Latency (one RTX PRO 6000 Blackwell, one request at a time; 1,000-row public board sample): median 291 ms, mean 652 ms, p80 871 ms.
- Third-party components: the executor derives from LOOM (Apache-2.0); the knowledge-store table is CC BY-SA 4.0. Notices are in the
  model repository.
- Licence of the model repository: Apache-2.0. Part of the training data is non-commercial or states no licence; the model card lists
  those sources.
