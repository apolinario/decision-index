# muria

Self-run Decision Index 0.3 public score **51.40** as scored (raw 63.16, breadth 49.63). With the 3 trained-on test items
below counted as wrong, it is **51.40**. The run is complete: 140,620 requests run, 140,178 scored after the official 442 exclusions,
140,178 answered, 0 unsupported, 0 errors. This is not the private Full score or a maintainer-verified rank.

- Model: https://huggingface.co/plantwaterer/muria/tree/f41f120a7e500a449066e5c3c72a1590b84e7179, a full fine-tune of google/gemma-4-12B-it with a
  calibrated last-layer head whose output is the answer: temperature-scaled option-letter logits on choice questions, plus a
  Bayesian last-layer residual on yes / no questions. Code is mirrored at https://github.com/treewaterer/muria.
- Results: https://huggingface.co/datasets/plantwaterer/muria-decision-index/tree/957077a309146396462ed2b6729c6b6db165c67c/runs/muria (run files as
  written by the kit's runner and scorer, unmodified).
- Kit: apolinario/decision-index @ 9eb2dbe2a358004c8782c66e40a83ac07b953fec, unchanged runner and scorer.
- Engine: `dikit_engine:DikitEngine`, in-process, from the model repository (it imports the other five files from the same repository):
  - `dikit_engine.py`, SHA-256 63bc2c878e8a4a615699a2ee4c20c679ba208c1cf2e2aabf5f547695317712b9;
  - `garas_vbll_engine.py`, SHA-256 dd1f6d55aa90e4b17c6b9b7b8b582fd610d6a830c4635529270499e5c1bb82f0;
  - `vbll_core.py`, SHA-256 b5ec4a496c608712bae15680f83b4de80c59385776eb8eec453a99c44024b0bd;
  - `fullft_mem.py`, SHA-256 31d02612c4bf896f7abe702caea61fb14c8dab307a393d3845942021abfb6c4f;
  - `garas_engine.py`, SHA-256 b35ee6482dc351aee48dca7cedc48e640e4efffda42a3fb9b648929e47ccd5d4;
  - `garas_core.py`, SHA-256 a7f4d8192d6b287b7c57d22ed7c33065a8d753650fd11e9bcf7f9e378b5c3a6e.

## Settings

```bash
REPO=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('plantwaterer/muria'))")
PYTHONPATH="$REPO" python -m decision_index run --engine dikit_engine:DikitEngine --option model="$REPO" --compact --out runs/muria
```

- **Precision:** bf16 weights and backbone, eager PyTorch (torch 2.14.1+cu130, transformers 5.12.1). The head runs on the CPU in float64.
- **Attention:** the model's full-attention layers attend bidirectionally within each prompt and its sliding-window layers causally
  (window 1,024). It is served with the `jev_mem` attention function in `fullft_mem.py` (registered with transformers' attention
  interface, PyTorch SDPA memory-efficient / flash kernels); the engine refuses transformers' stock attention, which would run the
  model causally.
- **Prompts:** one prompt per question. A choice question's options are shown in one fixed non-identity order derived from a hash
  of the request. Yes / no definitions, when the request provides them, are shown with the options.
- **Read-out:** one forward pass per prompt; the probabilities are the head's output from the hidden state and the option letters'
  logits.
- **More than 52 options:** handled in a second round over the first round's leaders.
- **Batching:** the prompts of one request run in packed rows of up to 32,768 tokens, each prompt with its own positions.
- **Hardware:** one GPU (bf16 weights 24 GB); fits one RTX PRO 6000 (96 GB).

## Run identity

- **Shards:** the full public suite ran as 8 row shards (row i to shard i mod 8), each on one GPU: 4 on NVIDIA H100 80GB HBM3
  (shards 0, 1, 2 and 7) and 4 on NVIDIA RTX PRO 6000 Blackwell (shards 3 to 6). bf16 results differ slightly between GPU types.
- **Runner:** each shard used the unchanged kit runner with resume.
- **Merge and scoring:** the shard result lines were concatenated unchanged into one `results.jsonl`; resume passes of the unchanged
  runner then ran 0 rows, and the unchanged scorer scored the file. The runner's `model` option pointed at a local copy of the model
  repository's files; `environment.json` records the engine files' md5s.

## Capacity limits

- **Context:** max_len is 32,768 tokens. A request that does not fit is reported as unsupported, and nothing is truncated.
- **Unsupported rows:** none.

## Training data overlap

- **Train splits used:** no test split of any benchmark was used to build the training data. The head's fit and calibration data
  include train-split rows of CLINC150, GSM8K, Amazon ESCI, BANKING77 and WinoGrande, and DocNLI units, some of which contain ANLI
  premise-hypothesis pairs; the training data include Lichess puzzles (a different dataset from ChessBench's).
- **Overlap found:** an audit of the training text, the head's data included, against the public suite found 3 ANLI test items that
  are near-duplicates of ANLI pairs inside DocNLI units in the head's fit data. Following the board's contamination rule, they are
  counted as wrong in the 51.40 above: `12:ANLI:ANLI:test_r2:5460dddd-0280-47bd-9148-1422b6729a4d`,
  `12:ANLI:ANLI:test_r2:f6a3b2f4-863e-41af-bde1-762da7822ebd`, `12:ANLI:ANLI:test_r3:d469aabc-dd2a-4bd7-b840-ab39764b6625`.

## Latency (our approximation)

From the run above: the kit runner's CUDA-synchronized per-request wall time on its four RTX PRO 6000 shards, one request at a time,
70,309 answered requests. This approximates the board's check, which uses a private sample.

| median | mean | 80th percentile |
|---|---|---|
| 48 ms | 209 ms | 169 ms |
