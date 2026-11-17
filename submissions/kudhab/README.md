# 𒋻𒇥 (kudhab)

Decision Index 0.3 public score: **73.96** (self-run; raw 80.03, breadth 73.54). 140,620 requests run, 140,178 scored, 403 unsupported
(over-length; 11 of them among the scored requests), 0 errors.

- Model: https://huggingface.co/plantwaterer/kudhab/tree/957137ea436f79d23ce6db6b10ba7707bf33ef58
- Results: https://huggingface.co/datasets/plantwaterer/kudhab-decision-index/tree/e12413e1e55f19c4733a091447c429d8c40ea70f/runs/kudhab
- Kit: apolinario/decision-index @ 9eb2dbe2a358004c8782c66e40a83ac07b953fec.
- Engine: `karpo_decision.serve.engine:DecisionIndexEngine` from machinepulse-ai/karpo-decision @ a6fa38bf4e8de9dbdb9491fb461b9b320a2e2118
  (Apache-2.0, by MachinePulse; an unmodified copy is in the model repository under `karpo-decision/`), in-process, view `joint` (the default), `--option max_length=32768`.

## Settings

```bash
REPO=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('plantwaterer/kudhab', revision='957137ea436f79d23ce6db6b10ba7707bf33ef58'))")
PYTHONPATH="$REPO/karpo-decision/src" python -m decision_index run --engine karpo_decision.serve.engine:DecisionIndexEngine \
    --option checkpoint="$REPO" --option max_length=32768 --compact --out runs/kudhab
```

- bf16, one GPU (fits one RTX PRO 6000 at max_length 32,768). Our runs: Python 3.12, torch 2.14.1+cu130, transformers 5.21.0, karpo-decision's plain
  attention path (without its optional `blackwell` kernels). Inference details are in the model card.
- Declared capacity limit: 32,768 tokens per sequence (`max_length`); a question that does not fit with its state is refused, never
  truncated (403 requests in the public run; the same 403 requests as in karpo-decision-31B's own published run).
- Latency (one RTX PRO 6000 Blackwell, one request at a time; 1,000-row public board sample): median 92 ms, mean 246 ms, p80 237 ms (max_length 32,768).
- Training: a fine-tune of a uniform average of four fine-tunes of a munku + Jeffan-31b-1710 merge, every stage trained to match the output probabilities of MachinePulseAI/karpo-decision-31B on our training questions; no karpo-decision weights. Jeffan-31b-1710 is used at its Apache-2.0 revision cebfa86c.
- Licence of the model repository: Apache-2.0. Part of the training data is non-commercial or states no licence; the model card lists
  those sources.
