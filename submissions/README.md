# Decision Index Leaderboard Submissions

This document records submitted candidate model evaluation runs for review and inclusion on the Decision Index leaderboard.

## Submissions Table

| Model Name | Submitter | Results Dataset | Engine / Commit | Hardware | Latency (Median) | Declared Capacity Limits |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DM-JEPA** | Danger Labs | [DangerLabs/decision-index-results](https://huggingface.co/datasets/DangerLabs/decision-index-results) (`runs/dm-jepa/scores.json`) | `dm-jepa` / [`HEAD`](https://github.com/DangerLabs/DM-JEPA) | 1x NVIDIA GeForce RTX 3060 (12GB) | **33.1 ms** | `max_state_length=2048`, `max_option_length=256` (no truncation, raises `Unsupported`) |

---

### Submission Details: DM-JEPA (Danger Labs)

- **Model Weights & Architecture Repository**: [DangerLabs/DM-JEPA](https://huggingface.co/DangerLabs/DM-JEPA)
- **Model Type**: Non-Autoregressive System 1 Joint Embedding Predictive Architecture (JEPA)
- **Evaluation Dataset**: [DangerLabs/decision-index-results](https://huggingface.co/datasets/DangerLabs/decision-index-results)
- **Results Folder**: `runs/dm-jepa/`
  - `scores.json`: Contains `"complete": true`, `"engine": "dm-jepa"`, and full metric breakdown.
  - `benchmark-summary.json`: Detailed per-benchmark native scoring summary.
  - `index.json`: Decision Index 0.2.1 calculation breakdown.
  - `environment.json`: Hardware, runtime environment, and declared capacity parameters.
  - `status.json`: Completed run status.
  - `results.jsonl.gz`: Full serialized response log.
- **Inference Speed**: Measured single-process synchronized in-process forward pass latency with median **33.1 ms** per request.
- **Compliance**:
  - No truncation: requests exceeding declared capacity raise `Unsupported`.
  - No prompt tuning: uniform formatting across evaluation benchmarks.
  - Full valid probability distribution over all criteria options.
