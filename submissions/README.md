# Decision Index submissions

| Model | Model repository | Public 0.3 results | Engine and source | Hardware | Inference settings | Capacity declaration |
| --- | --- | --- | --- | --- | --- | --- |
| BRREV-27B v0.1.0 | [Private Hub model](https://huggingface.co/anuragranj/brrev-27b-v0.1.0-eval) — read-only maintainer access to be arranged on verified request | [Complete public 0.3 run: 57.81](https://huggingface.co/datasets/anuragranj/brrev-decision-index-results/blob/main/runs/brrev-27b-v0.1.0-kit-0.3-h100x2-b16k-gsm-only-20261009/scores.json) | `brrev.decision_index:BrrevEngine`, private BRREV engine commit `54f9f855d36837dd848a70f071d1a80640b04ab0`; kit commit `9eb2dbe2a358004c8782c66e40a83ac07b953fec` | One GCP spot node, NVIDIA H100 80GB × 2, model parallel | `temperature=1.0`, `device=cuda:0`, `model_parallel_gpus=2`, `max_batch_tokens=16384`, independent per-question probabilities, no answer-option filtering or truncation | Questions over 16,384 tokens declared unsupported; all other questions untruncated. 140,178 scoreable requests completed: 140,176 answered, two unsupported, zero runtime errors. |

This submission uses the 0.2.1 run's unmodified response rows and adds only
the 2,638 rebuilt GSM8K requests required by 0.3. The kit re-scores the
complete response file under edition 0.3; the earlier 0.2.1 score is not the
0.3 public score. The private model repository contains the inference-only
checkpoint and exact inference code; no weights or BRREV source are public.
The result rows and score files are byte-for-byte unmodified; the public
`environment.json` discloses a redaction of private checkpoint paths and
weight hashes while retaining hardware and capacity limits. The original
environment file and its SHA-256 remain available for private reviewer
verification. The Full score is pending the maintainers' private evaluation.
