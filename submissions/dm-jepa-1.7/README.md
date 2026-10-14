# DM-JEPA 1.7 (`DMBertJEPA17`) — Danger Labs

Self-run **Decision Index 0.3** public score **90.66** (Gold ★ weighted 0.3 public index; **93.62** balanced 5-area raw index) and **Vision 0.3** public score **74.22** at **25.38 ms** median forward latency per request. The run is complete (`"complete": true` in `scores.json`): 140,620 Edition 0.3 requests run, 140,178 scored after the official 442 exclusions (109,759 base with rebuilt `v0.3-gsm8k:` rows and ForecastBench retired + 30,419 added), 0 unsupported, 0 errors.

- **Model repository:** [https://huggingface.co/DangerLabs/dm-jepa-1.7/tree/9f2f6f85a2923c0f2307cc966b7ead36723e96dc](https://huggingface.co/DangerLabs/dm-jepa-1.7/tree/9f2f6f85a2923c0f2307cc966b7ead36723e96dc) (`DMBertJEPA17`, 349,030,286 parameters: 16-layer `Cart-BB` 205.36M backbone with `d_model=1536`, 24 heads, `BlockButterflyLinear` $P=32$, `DFlashThoughtDrafter`, 6-step `AdaptiveEnergyPredictor17`, `RiemannianManifoldScorer` + `HJEPAAbstractProjection`, and `VisionEncoderJEPA17`).
- **Complete public results:**
  - **Text 0.3:** [https://huggingface.co/datasets/DangerLabs/decision-index-results/tree/cc927b7584172189cf281cb560ff79c9bfc3c67d/runs/dm-jepa-1.7](https://huggingface.co/datasets/DangerLabs/decision-index-results/tree/cc927b7584172189cf281cb560ff79c9bfc3c67d/runs/dm-jepa-1.7) (`scores.json`, `index.json`, `benchmark-summary.json`, `environment.json`, `status.json`, `results.jsonl.gz`).
  - **Vision 0.3:** [https://huggingface.co/datasets/DangerLabs/decision-index-results/tree/cc927b7584172189cf281cb560ff79c9bfc3c67d/runs/dm-jepa-1.7-vision](https://huggingface.co/datasets/DangerLabs/decision-index-results/tree/cc927b7584172189cf281cb560ff79c9bfc3c67d/runs/dm-jepa-1.7-vision) (`vision-scores.json`, `environment.json`, `status.json`).
- **Kit:** `apolinario/decision-index` @ `f5a6e4f3e072cd5ef33db80dc0a7ff9bd2d6d045`.
- **Engine:** `dm_jepa_1_7_engine:DMJEPA17Engine` (included in this folder and in the model repository):
  - `dm_jepa_1_7_engine.py`, SHA-256 `5177a4ff3c2bb373e4bf107139fbe0026edb7dbf3b11af5bd60484b03cd9c36d`.

## Settings & Reproduction

```bash
# 1. Text Suite (Edition 0.3)
PYTHONPATH=submissions/dm-jepa-1.7 python -m decision_index pipeline \
  --edition 0.3 \
  --engine dm_jepa_1_7_engine:DMJEPA17Engine \
  --option model=DangerLabs/dm-jepa-1.7 \
  --option dtype=bfloat16 \
  --option max_state_length=32768 \
  --option max_option_length=512 \
  --compact --out runs/dm-jepa-1.7

# 2. Vision Suite (Edition 0.3)
PYTHONPATH=submissions/dm-jepa-1.7 python -m decision_index vision pipeline \
  --dir work/vision \
  --engine dm_jepa_1_7_engine:DMJEPA17Engine \
  --option model=DangerLabs/dm-jepa-1.7 \
  --option max_pixels=1638400 \
  --workers 1 --out runs/dm-jepa-1.7-vision
```

- **Precision:** `bfloat16` on CUDA (`torch.autocast("cuda", dtype=torch.bfloat16)`), deterministic non-autoregressive forward pass (`model.eval()`, `adaptive_early_exit=False`).
- **Read-out:** Single non-autoregressive forward pass encoding `state` once through `DMBertBackbone17` and all candidate `criteria` in parallel through `criteria_encoder`, followed by 6-step latent rollout (`AdaptiveEnergyPredictor17`) and two-level geodesic scoring:
  $$\ell_k = (1 - \alpha)\,\ell_k^{(L_1,\text{Riemannian})} + \alpha\,\ell_k^{(L_2,\text{H-JEPA})}$$
- **No truncation & no option filtering:** Requests whose tokenized state exceeds `max_state_length=32768` or any option exceeds `max_option_length=512` raise `decision_index.engines.Unsupported` without truncating. Every option in `criteria` receives a finite softmax probability summing to `1.0`, validated via `decision_index.engines.validate`.
- **Vision 1.6 MP native resolution rule:** Images are loaded at native resolution capped at `1,638,400` pixels (`1.6 MP`) and encoded via `VisionEncoderJEPA17` (`patch_conv` + 16-region spatial bounding-box stream `bbox_encoder`) gated by `tanh(multimodal_gate)`.

## Capacity Limits

- **Context:** `max_state_length = 32,768` RoPE tokens; `max_option_length = 512` tokens per option.
- **Unsupported rows:** `0` unsupported requests across the Edition 0.3 suite.

## Latency

Single-process device-synchronized forward latency measured through the engine:

| Median | Mean | 80th Percentile |
|---:|---:|---:|
| **25.38 ms** | **28.44 ms** | **31.20 ms** |
