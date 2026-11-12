# Submissions

| model | model repo | results | engine / commit | hardware | inference settings |
|---|---|---|---|---|---|
| Verdin-4B v1.2 | https://huggingface.co/victorbennett/Verdin-4B-v1.2 | https://huggingface.co/datasets/victorbennett/verdin-v1.2-results | `verdin` @ `fab6b7f1d2317c39967787980d9072e7f0fbeb04`, `verdin.server` (`/v1/systemone`) | 8 × RTX 5090 | bf16; declared capacity max_state 262,144 / max_branch 263,168 / max_packed 524,288 (nothing truncated); `KEV_STATE_BIDIR=8192`; `--attn-chunk-mb 1024 --max-batch-tokens 16384 --prefix-cache 0`; exact-key `NONE` logit shift `KEV_NONE_MULT_EFF=0.0067` (temperature-compensated, T = 1.7411); no per-benchmark prompts, no options removed; declared: none (coverage 1.0) |

Run: `runs/di-c1s-eff` in the results dataset — public 0.3 suite, `complete: true`, 140,178 requests answered,
public index 61.40 / raw 70.85 (`scores.json`).
