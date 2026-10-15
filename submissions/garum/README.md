# garum

Self-run Decision Index 0.3 public score **58.36** as scored (raw 68.61, breadth 57.43). With the 19 trained-on test items below
counted as wrong, it is **58.34**. The run is complete: 140,620 requests run, 140,178 scored after the official 442 exclusions,
139,869 answered, 309 unsupported (over 32,768 tokens; no truncation), 0 errors. This is not the private Full score or a
maintainer-verified rank.

- Model: https://huggingface.co/plantwaterer/garum/tree/a463e943dd7d022c7332499490e3c94fe8119397, a LoRA adapter for
  google/gemma-4-31B-it with a calibrated Bayesian last-layer head (the head's output is the answer). Code is mirrored at
  https://github.com/treewaterer/garum.
- Results: https://huggingface.co/datasets/plantwaterer/garum-decision-index/tree/d6d44ccad5995184b1b69413f8aa5468fdda4c9a/runs/garum (run files as written by the
  kit's runner and scorer, unmodified).
- Kit: apolinario/decision-index @ 9eb2dbe2a358004c8782c66e40a83ac07b953fec, unchanged runner and scorer.
- Engine: `garum_engine:GarumEngine`, in-process, from the model repository (it imports `modeling_garum.py`, `garum_core.py`
  and `garum_head.py` from the same repository):
  - `garum_engine.py`, SHA-256 49e16de65c635252481a9fc6cf2d777dfc24b0fb4905e1fbfdebd8d160cffc4f;
  - `modeling_garum.py`, SHA-256 6aaaf01dd2fa68bb2afe26e3be8e8329f0a4dc779488866ec803c73c75f0f3da;
  - `garum_core.py`, SHA-256 8d0f3f1c98b233d3f8a11d4ca748cbde2c5295fe1dd58029b0fc744821f873e1;
  - `garum_head.py`, SHA-256 a327645025c5e88f78984b74de7adef7f3e6aae47e0801f69fe0fa2ecfab44b5.

## Settings

```bash
python -m decision_index run --engine garum_engine:GarumEngine \
  --option base=google/gemma-4-31B-it --option adapter=plantwaterer/garum --compact --out runs/garum
```

- **Precision:** bf16, eager PyTorch, transformers' default attention (SDPA) (transformers 5.12.1, peft 0.21.2, torch 2.13.0).
  The adapter is loaded with `autocast_adapter_dtype=False`, so the LoRA update runs in bf16.
- **Prompts:** one prompt per question. A choice question's options are shown in one fixed non-identity order derived from a hash
  of the request. Yes / no definitions, when the request provides them, are shown with the options.
- **Read-out:** one forward pass per prompt; the probabilities are the head's output from the hidden state and the option letters'
  next-token logits.
- **More than 52 options:** handled in a second round over the first round's leaders.
- **Batching:** the prompts of one request that have the same token length run together, unpadded, up to 24,576 tokens; every
  other prompt runs alone.

## Run identity

- **Shards:** the full public suite ran as 8 row shards (row i to shard i mod 8), each on one GPU: 3 on NVIDIA H100 80GB HBM3
  (shards 0, 1 and 7) and 5 on NVIDIA RTX PRO 6000 Blackwell (shards 2 to 6).
- **Mixed hardware:** bf16 results differ slightly between the two GPU types. For Garas, re-scoring its RTX shards on H100 flipped
  0.85% of 120,556 scored decisions and left the index unchanged at the kit's 0.01 precision; we have not repeated that check for garum.
- **Model files:** the runner's `adapter` option pointed at a local copy of the model repository's files (adapter, head and code);
  `environment.json` records the engine files' md5s, which match the repository.
- **Runner:** each shard used the unchanged kit runner with resume.
- **Merge and scoring:** the shard result lines were concatenated unchanged into one `results.jsonl`, then one resume pass of the
  runner ran over the merged directory (0 rows run) and the unchanged scorer scored it.

## Capacity limits

- **Context:** max_len is 32,768 tokens. A request that does not fit is reported as unsupported, and nothing is truncated.
- **Unsupported rows:** there are 309: MMLU-Pro 304 and iSarcasmEval 5. They count as wrong.

## Training data overlap

- **Train splits used:** training used the train splits of BANKING77, CLINC150 and GSM8K, plus WinoGrande train, Amazon ESCI
  train queries, Lichess puzzles (a different dataset from ChessBench's) and ANLI training-split pairs that are part of DocNLI.
  No test split of any benchmark was used.
- **Overlap found:** an audit of the training text against the public suite found 19 test items that are exact or near-exact
  duplicates of train-split utterances. Following the board's contamination rule, they are counted as wrong in the 58.34
  above:
  - `4:BANKING77:BANKING77:test:` 554, 677, 727, 742, 747, 751, 754, 1432, 1735, 1754, 1993, 2081, 2211, 2440, 2644, 2651, 2987,
    3070;
  - `5:CLINC150+OOS:CLINC150+OOS:test:1591`.

## Latency (our approximation)

These numbers come from 1 x RTX PRO 6000, one request at a time, measured through the kit's runner. The sample is our own 760 rows
of the public suite: 750 timed plus 10 warm-up. This approximates the board's check, which uses a private sample.

| median | mean | 80th percentile |
|---|---|---|
| 115 ms | 492 ms | 424 ms |
