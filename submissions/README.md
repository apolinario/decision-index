# Model submissions

Author-run results pending maintainer validation.

| Model | Index | Results | Engine / revision | Hardware / capacity | Notes |
|---|---|---|---|---|---|
| Wald-Q4B v1.1 (22D0-f7) | 54.59 (0.2.1; complete) | [scores / run](https://huggingface.co/datasets/Harry19081/Wald-Q4B-decision-index-results/blob/805716601b2466be324ed6716407b4c3d9267faa/runs/wald-q4b-22d0-f7-full021/scores.json) | [weights + reference engine](https://huggingface.co/Harry19081/Wald-4B/tree/50f94ecd5d6e7e8459e9c198ef811bfaba12a3fe); http / vLLM 0.30.0 | 1× RTX PRO 6000, BF16, 131,072-token context, no truncation | effort high; serial latency not yet validated (our 32-request median 821 ms) |
