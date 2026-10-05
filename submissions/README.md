# Decision Index Leaderboard Submissions

This document records submitted candidate model evaluation runs for review and inclusion on the Decision Index leaderboard.

## Submissions Table

| Model Name | Submitter | Results Dataset | Engine / Commit | Hardware | Latency (Median) | Declared Capacity Limits |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DM-JEPA** | Danger Labs | [DangerLabs/decision-index-results](https://huggingface.co/datasets/DangerLabs/decision-index-results) (`runs/dm-jepa/scores.json`) | `dm-jepa` / `effc5c2` | 1x NVIDIA GeForce RTX 3060 (12GB) | **33.1 ms** | `max_state_length=2048`, `max_option_length=256` (no truncation, raises `Unsupported`) |

---

### Submission Details: DM-JEPA (Danger Labs)

- **Model Weights & Architecture Repository**: [DangerLabs/DM-JEPA](https://huggingface.co/DangerLabs/DM-JEPA)
- **Model Type**: Non-Autoregressive System 1 Joint Embedding Predictive Architecture (JEPA)
- **Evaluation Dataset**: [DangerLabs/decision-index-results](https://huggingface.co/datasets/DangerLabs/decision-index-results)
- **Results Folder**: `runs/dm-jepa/`
  - `scores.json`: Contains `"complete": true`, `"completed": 150759`, `"decision_index": 18.37`, `"engine": "dm-jepa"`, and full 44-benchmark metric breakdown.
  - `benchmark-summary.json`: Detailed per-benchmark native scoring summary across all 44 datasets.
  - `index.json`: Decision Index 0.2.1 calculation breakdown (Balanced Skill: 18.37, Balanced Raw: 37.62, Breadth Skill: 17.86).
  - `environment.json`: Hardware, runtime environment, and declared capacity parameters.
  - `status.json`: Completed run status across all 150,759 requests.
  - `results.jsonl.gz`: Full serialized response log (39.89 MB, 150,759 evaluated rows).
- **Inference Speed**: Measured single-process synchronized in-process forward pass latency with median **33.1 ms** per request.
- **Compliance**:
  - No truncation: requests exceeding declared capacity raise `Unsupported`.
  - No prompt tuning: uniform formatting across evaluation benchmarks.
  - Full valid probability distribution over all criteria options.
