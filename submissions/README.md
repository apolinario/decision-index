# Model submissions

| Model | Model repository | Complete public results | Engine and code revision | Hardware | Inference settings |
| :--- | :--- | :--- | :--- | :--- | :--- |
| PICK-150M V5 | [lCREEl/pick-150m](https://huggingface.co/lCREEl/pick-150m) — weights `9980b9adcc14331a9b9d44bb936aca0f7deb221d` | [scores.json](https://huggingface.co/datasets/lCREEl/pick-v5-decision-index-submission/blob/76d789d19917c9c3a0916427d90caa5a3b4ce2e9/runs/pick-150m-v5/scores.json): public index **32.09**, edition 0.3, `complete: true`, 140,178 scoreable requests | `pick_v5.engine:PickV5Engine`; [source/wheel](https://huggingface.co/lCREEl/pick-150m/tree/f1635cf8868b3359c0ea78cd7d0110e7bbe6b555/decision-index), commit `f1635cf8868b3359c0ea78cd7d0110e7bbe6b555`; kit `f5a6e4fd85fed05e77878dfc4966d0568bfb45ec` | Free Colab Tesla T4; three resumed sessions, identical inference fingerprint | fp32, native CUDA TF32, no compilation, one request at a time, 7,999 packed tokens and forward token budget, shared-state packing, native probabilities, no benchmark-specific calibration |

## PICK-150M V5 details

- [Reproduction, full settings, development-data disclosure and original result files](https://huggingface.co/datasets/lCREEl/pick-v5-decision-index-submission).
- Final V5 stage-2-best model, 149,015,809 parameters. No weights, architecture or training changes for this submission.
- All questions and options are retained. Native deterministic recursive bisection handles multi-question requests that do not fit together; a branch that cannot fit refuses the whole request without truncation.
- 140,166 successful, 12 capacity refusals, 0 runtime errors, no pending scoreable requests. Refusals: ToolRet 5, ContractNLI 2, POP909 4, HLE 1. The untouched archive includes all 442 officially excluded rows as well.
- Prior development used related public task families and sampled benchmark feedback; no blanket unseen-data or semantic-decontamination claim is made.
- An independent canonical-suite re-score matches the submitted results exactly. [Audit and SHA256 values](https://huggingface.co/datasets/lCREEl/pick-v5-decision-index-submission/blob/main/final-run-audit.json).
- Private Full-score components and the official RTX PRO 6000 latency check are requested; T4 timings are not a substitute for that check.
