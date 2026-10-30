# Model submissions

| Model | Edition | Public Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [BaseDecision-Dyno](https://huggingface.co/onlyaady/BaseDecision-Dyno/tree/main) | 0.3 | 53.83 | [scores.json](https://huggingface.co/onlyaady/BaseDecision-Dyno/blob/main/scores.json) | [dyno_submission_engine:DynoSubmissionEngine](https://huggingface.co/onlyaady/BaseDecision-Dyno/blob/main/dyno_submission_engine.py); kit `25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d` | NVIDIA RTX PRO 6000 Blackwell MIG 2g.48gb; independent evaluation workers; BF16 backbone, FP32 readout | Text only; choice/noul; 16,384 rendered tokens per question including repetition; 2–255 choice options; no truncation; 162 unsupported |