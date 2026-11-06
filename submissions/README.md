# Submissions

| model | model repo | results | engine / code | hardware | settings |
|---|---|---|---|---|---|
| decy-preview | [PNPM244/decy-preview](https://huggingface.co/PNPM244/decy-preview) @ 287ebe7 | [PNPM244/decy-preview-decision-index/runs/decy-preview](https://huggingface.co/datasets/PNPM244/decy-preview-decision-index/tree/main/runs/decy-preview) (public 0.3: 68.03, complete) | kit 0.3 (62d2f51), in-process `decy:Decy` with `noncausal=1`, `layout=state` (`decy.py` + `bidir.py` in the model repo) | 4x H200 (4 processes, 1 per GPU) | bf16; full-attention layers bidirectional over the prompt; state-first layout; one forward pass per question, no generated tokens; softmax (temperature 1) over the option-label logits with the model's logit soft-capping (30); nothing truncated; 0 errors, 0 unsupported |
