# Model submissions

| Model | Model repository | Complete public results | Engine / hardware / settings |
|---|---|---|---|
| Qwen Decision 27B noncausal LoRA (step 1533) | [Model and inference code](https://huggingface.co/jsaurabh/qwen-decision-27b-noncausal-lora/tree/b2eefb929a883136b3bd56ce70999fdac656ee8a) | [Decision Index 0.3 scores.json](https://huggingface.co/datasets/jsaurabh/qwen-decision-27b-decision-index-0.3/blob/11a97b664e02ceb8428cdc1e42a2ce7905c152b3/runs/noncausal-step1533/scores.json) | HTTP System One engine, kit `9eb2dbe2a358004c8782c66e40a83ac07b953fec`; one RTX PRO 6000 Blackwell 96 GB; exact settings and disclosures below. |

## Qwen Decision 27B noncausal LoRA

Independent AutoJev-style LoRA/readout experiment, not the official Perplexity weights. Public index **57.70**, raw index **68.26**, `complete: true`; all 140,178 scoreable requests accounted for (140,148 supported, 30 unsupported, no errors). The untouched compact results are alongside `scores.json`. The runner processed 140,620 requests, including 442 excluded by official scoring.

### Reproduction

The pinned model repository contains weights, processor, exact evaluated `serve_checkpoint_original.py`, a portable `serve_checkpoint.py` whose request-handler AST is identical, pinned requirements and installation commands. Start the portable server, then:

```bash
python -m decision_index pipeline --engine http \
  --option base_url=http://127.0.0.1:8765 \
  --option model=our-noncausal-27b-step1533 \
  --option timeout=3600 --suite-dir /path/to/suite-0.3 \
  --out runs/noncausal-step1533 --compact
```

- Base: `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Source reference: `perplexity-ai/pplx-decider-v1.1-27b`, revision `3b45dead91dfa6d95aad6b95764a606fab2bf7a6`; exact locally modified inference source is bundled.
- Noncausal full SDPA attention, native recurrent linear-attention mask, last-token pooling, dedicated 255-way readout, LoRA rank 16 / alpha 32 / dropout 0.05.
- Saved calibration temperature `1.0862525693910237`; no fitting on Decision Index. Unchanged decision prompt construction across benchmarks.
- **Capacity:** `choice` and `noul`, at most255 options, 8192 tokens per complete question branch, batches of 4 questions, no truncation. Excess capacity is explicitly rejected.
- Single-process loopback HTTP server. No `causal_conv1d` installed; correct PyTorch fallback used. Dependencies and source hashes are in the model repository.
- The full run reused 58,142 earlier predictions produced by the same frozen checkpoint and protocol.

### Training disclosure and limitations

48,847 training rows,1533 optimizer steps, effective batch 32, training context 512, learning rate5e-5. The mix combines a deterministic per-suite text subset of the archived general public decision corpus with synthetic banking gap curriculum. Development and temperature splits each used512 rows. Data hashes and settings are in `decision_config.json`.

**Overlap with Decision Index public data has not been fully audited; no contamination-free claim is made.** Internal family separation across our training/dev/calibration splits does not establish independence from public benchmarks. Base-model pretraining overlap is unknown. The checkpoint was frozen before this evaluation, with no benchmark prompt tuning or calibration changes.

Measured public-run latency: median 105.9 ms, mean 383.2 ms, p95 1565.3 ms. These are not the maintainers' independent latency-gate measurements. We request review and private evaluation; no Full score or leaderboard rank is claimed.
