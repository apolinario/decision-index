# Submissions

| model | model repo | results | engine / commit | hardware | inference settings |
|---|---|---|---|---|---|
| Verdin-4B v1.2 | https://huggingface.co/victorbennett/Verdin-4B-v1.2 | https://huggingface.co/datasets/victorbennett/verdin-v1.2-results | `verdin` @ `5d6ed3249390750aa46445144c6e5dab4d40f66c`, `verdin.server` (`/v1/systemone`) | 8 × RTX 5090 | bf16; declared capacity max_state 262,144 / max_branch 263,168 / max_packed 524,288 (nothing truncated); `KEV_STATE_BIDIR=8192`; `--attn-chunk-mb 1024 --max-batch-tokens 16384 --prefix-cache 0`; exact-key `NONE` logit shift `KEV_NONE_MULT_EFF=0.0067` (temperature-compensated, T = 1.7411); no per-benchmark prompts, no options removed; declared: none (coverage 1.0) |

Run: `runs/di-c1s-soup-eff` in the results dataset (s0+s1 LoRA soup) — public 0.3 suite, `complete: true`, 140,178 requests answered,
public index 61.84 / raw 71.16 (`scores.json`).
