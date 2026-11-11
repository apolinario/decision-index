# Submissions

| model | model repo | results | engine / code | hardware | settings |
|---|---|---|---|---|---|
| GS1-0.8B | [bigai/GS1-0.8B](https://huggingface.co/bigai/GS1-0.8B) @ 2bf1cac | [runs/GS1-0.8B](https://huggingface.co/datasets/bigai/GS1-Decision-Index-Public-Results/tree/1c3b2b438b1e46ad9d4d315d31cecd19ee9be356/runs/GS1-0.8B) ([scores.json](https://huggingface.co/datasets/bigai/GS1-Decision-Index-Public-Results/blob/1c3b2b438b1e46ad9d4d315d31cecd19ee9be356/runs/GS1-0.8B/scores.json); public 0.3: 56.12; complete) | [GS1](https://github.com/bigai-nlco/GS1); [native reference adapter](https://huggingface.co/datasets/bigai/GS1-Decision-Index-Public-Results/blob/1c3b2b438b1e46ad9d4d315d31cecd19ee9be356/code/gs1_release_engine.py) @ 1c3b2b4; kit 0.3 @ 25c6ec7 | Newly evaluated Public requests: one node, 6 × NVIDIA GeForce RTX 4090 (48 GiB reported VRAM/device); independent single-GPU replicas; see shared hardware details below | BF16 compute; original stored dtypes; `reference_batching=false`; cache off; seed 41; no truncation or option filtering; shared limits below |
| GS1-2B | [bigai/GS1-2B](https://huggingface.co/bigai/GS1-2B) @ 9c37499 | [runs/GS1-2B](https://huggingface.co/datasets/bigai/GS1-Decision-Index-Public-Results/tree/1c3b2b438b1e46ad9d4d315d31cecd19ee9be356/runs/GS1-2B) ([scores.json](https://huggingface.co/datasets/bigai/GS1-Decision-Index-Public-Results/blob/1c3b2b438b1e46ad9d4d315d31cecd19ee9be356/runs/GS1-2B/scores.json); public 0.3: 62.22; complete) | as above | Newly evaluated Public requests: one node, 6 × NVIDIA GeForce RTX 4090 (48 GiB reported VRAM/device); independent single-GPU replicas; see shared hardware details below | as above |
| GS1-4B | [bigai/GS1-4B](https://huggingface.co/bigai/GS1-4B) @ ddc2bb8 | [runs/GS1-4B](https://huggingface.co/datasets/bigai/GS1-Decision-Index-Public-Results/tree/1c3b2b438b1e46ad9d4d315d31cecd19ee9be356/runs/GS1-4B) ([scores.json](https://huggingface.co/datasets/bigai/GS1-Decision-Index-Public-Results/blob/1c3b2b438b1e46ad9d4d315d31cecd19ee9be356/runs/GS1-4B/scores.json); public 0.3: 68.97; complete) | as above | Newly evaluated Public requests: one node, 6 × NVIDIA GeForce RTX 4090 (48 GiB reported VRAM/device); independent single-GPU replicas; see shared hardware details below | as above |
| GS1-9B | [bigai/GS1-9B](https://huggingface.co/bigai/GS1-9B) @ 50cab2c | [runs/GS1-9B](https://huggingface.co/datasets/bigai/GS1-Decision-Index-Public-Results/tree/1c3b2b438b1e46ad9d4d315d31cecd19ee9be356/runs/GS1-9B) ([scores.json](https://huggingface.co/datasets/bigai/GS1-Decision-Index-Public-Results/blob/1c3b2b438b1e46ad9d4d315d31cecd19ee9be356/runs/GS1-9B/scores.json); public 0.3: 70.45; complete) | as above | Newly evaluated Public requests: one node, 6 × NVIDIA GeForce RTX 4090 (48 GiB reported VRAM/device); independent single-GPU replicas; see shared hardware details below | as above |

## GS1 shared evaluation details

- Deployment/inference code: [bigai-nlco/GS1](https://github.com/bigai-nlco/GS1). The exact [runtime snapshot](https://huggingface.co/datasets/bigai/GS1-Decision-Index-Public-Results/tree/1c3b2b438b1e46ad9d4d315d31cecd19ee9be356/code/runtime) and [evaluation adapter](https://huggingface.co/datasets/bigai/GS1-Decision-Index-Public-Results/blob/1c3b2b438b1e46ad9d4d315d31cecd19ee9be356/code/gs1_release_engine.py) are also available in the public results repository.
- Engine: `gs1_release_engine:GS1Engine`, native reference path. Each run has 140,178 successful requests and 312,320 decisions across 43 displayed benchmarks (37 counted), with `complete: true`, 100% coverage, zero errors, and zero unsupported/pending requests.
- Inference: BF16 compute with original persisted checkpoint dtypes, `reference_batching=false`, cache disabled, seed 41. Limits: 32,768 context tokens, 131,072 request tokens, 512 questions, 5,120 branches, and 32,768 batch tokens. No input truncation, option filtering, benchmark-specific prompts, or weight changes.
- Reproduction and re-scoring instructions: [dataset card](https://huggingface.co/datasets/bigai/GS1-Decision-Index-Public-Results/blob/1c3b2b438b1e46ad9d4d315d31cecd19ee9be356/README.md). Per-request probabilities, original scoring outputs, model hashes, and parity evidence are included. Upstream kit: `25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d`; adapter SHA-256: `0ecc8e0127e62a4d8faf26c2ed511c9f69303a2a415be17c9d2b3ad111c8d71d`.
- Private paths and infrastructure-identifying metadata were redacted. Predictions, probabilities, and scoring outputs are unchanged. Other infrastructure-identifying metadata remains redacted.
- **Hardware:** The newly evaluated Public requests ran on a single node with six NVIDIA GeForce RTX 4090 GPUs (48 GiB reported VRAM per device), using independent single-GPU replicas with two CPU threads per replica and BF16 computation while preserving original checkpoint storage dtypes; some GS1-0.8B requests shared a GPU with another inference workload, so the recorded timings do not represent isolated single-request latency.
- Request: validate the public runs and evaluate all four checkpoints on the maintainer-run private tests and latency qualification for the Text leaderboard. No Full score or Vision result is claimed.

## Resumed records and pinned reproduction

**Resumed records:** GS1-0.8B, GS1-2B, and GS1-4B each reuse 110,087 historical predictions; GS1-9B reuses 110,085 (two records missing original start timestamps were rerun). The remaining 30,091 requests per model, or 30,093 for 9B, were newly evaluated. As documented in the [evaluation method and resumed-record disclosure](https://huggingface.co/datasets/bigai/GS1-Decision-Index-Public-Results/blob/1c3b2b438b1e46ad9d4d315d31cecd19ee9be356/README.md#evaluation-method-and-resumed-records), reuse required matching every release tensor and every input hash and label, plus release-checkpoint parity checks on 73 representative/longest requests covering all 37 index benchmarks; the reported probability differences were zero. Original responses, probabilities, model identifiers, and measured durations are preserved; derived legacy completion timestamps are disclosed in each run's `environment.json`. We request maintainer review of these resumed runs.

**Reproduce the pinned models:** use Python 3.11+ and a compatible CUDA environment; reconstruct and verify the public suite following the [dataset card](https://huggingface.co/datasets/bigai/GS1-Decision-Index-Public-Results/blob/1c3b2b438b1e46ad9d4d315d31cecd19ee9be356/README.md#source-alignment), including its documented historical SGD-builder exception. The following pins both the published code snapshot and all four model revisions, then passes local checkpoint directories to the adapter (which has no separate `revision` option).

```bash
hf download bigai/GS1-Decision-Index-Public-Results --repo-type dataset \
  --revision 1c3b2b438b1e46ad9d4d315d31cecd19ee9be356 \
  --include 'code/*' --local-dir gs1-evaluation
cd gs1-evaluation
python -m pip install -e code/runtime -e 'code/decision-index[rebuild]'
export PYTHONPATH="$PWD/code:$PWD/code/runtime:$PWD/code/decision-index"
# Set this to the absolute path of the reconstructed, verified public suite.
SUITE_DIR=/absolute/path/to/suite-0.3
python code/stage_inputs.py --suite-dir "$SUITE_DIR" --out text-inputs.jsonl
while read -r model revision; do
  hf download "bigai/$model" --revision "$revision" --local-dir "models/$model"
  python -m decision_index pipeline --edition 0.3 \
    --suite-dir "$SUITE_DIR" --compact --rows "$PWD/text-inputs.jsonl" \
    --engine gs1_release_engine:GS1Engine \
    --option "model=$PWD/models/$model" --option max_context=32768 \
    --option seed=41 --out "reproduced/$model"
done <<'MODEL_REVISIONS'
GS1-0.8B 2bf1cace113e9b839f7a19843b91b6cf32c446c4
GS1-2B 9c3749961fb22ed667f4ad7a093f82c29bdbf0be
GS1-4B ddc2bb85ecba2131ddc6a6a370123cf1664b44eb
GS1-9B 50cab2ce35ca14ae318bb152e2396fac94f6b596
MODEL_REVISIONS
```
