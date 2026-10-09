# garas

Self-run Decision Index 0.3 public score **56.59** as scored (raw 67.38, breadth 55.67). With the 19 trained-on test items below
counted as wrong, it is **56.57**. The run is complete: 140,620 requests run, 140,178 scored after the official 442 exclusions,
139,869 answered, 309 unsupported, 0 errors. This is not the private Full score or a maintainer-verified rank.

- Model: https://huggingface.co/plantwaterer/garas/tree/505bdd3d677c9f1ef58e55766b354a87cecf81c8, a LoRA adapter for
  google/gemma-4-31B-it. Code is mirrored at https://github.com/treewaterer/garas.
- Results: https://huggingface.co/datasets/plantwaterer/garas-decision-index/tree/557760894aa2a50208a32072d9daa27ed550122e/runs/garas
  (run files as written by the kit's runner and scorer, unmodified).
- Kit: apolinario/decision-index @ 9eb2dbe2a358004c8782c66e40a83ac07b953fec, unchanged runner and scorer.
- Engine: `garas_engine:GarasEngine`, in-process, from the model repository:
  - `garas_engine.py`, SHA-256 c9eac62e0f64545e48a3aa08017300f0639b80c9f1c2ca104605a5a696c4b648;
  - `garas_core.py`, SHA-256 a7f4d8192d6b287b7c57d22ed7c33065a8d753650fd11e9bcf7f9e378b5c3a6e.

## Settings

```bash
python -m decision_index run --engine garas_engine:GarasEngine \
  --option base=google/gemma-4-31B-it --option adapter=plantwaterer/garas \
  --option max_len=32768 --option tokens_per_batch=32768 --compact --out runs/garas
```

- **Precision:** bf16, eager PyTorch (transformers 5.12.1, peft 0.21.2, torch 2.13.0). The adapter is loaded with
  `autocast_adapter_dtype=False`, so the LoRA update runs in bf16.
- **Prompts:** one prompt per question. A choice question's options are shown in one fixed non-identity order derived from a hash
  of the request.
- **Read-out:** one forward pass per prompt; the probabilities come from the option letters' next-token logits.
- **More than 52 options:** handled in a second round over the first round's leaders.
- **Batching:** the prompts of one request run in right-padded batches of up to 32,768 tokens.

## Run identity

- **Shards:** the full public suite ran as 8 row shards (row i to shard i mod 8), each on one GPU: 5 on NVIDIA H100 80GB HBM3 and 3
  on NVIDIA RTX PRO 6000 Blackwell.
- **Runner:** each shard used the unchanged kit runner with resume; no shard failed or was restarted.
- **Merge and scoring:** the shard result lines were concatenated unchanged into one `results.jsonl`, then one resume pass of the
  runner ran over the merged directory (0 rows run) and the unchanged scorer scored it.

## Capacity limits

- **Context:** max_len is 32,768 tokens. A request that does not fit is reported as unsupported, and nothing is truncated.
- **Unsupported rows:** there are 309: MMLU-Pro 304 and iSarcasmEval 5. They count as wrong.

## Training data overlap

- **Train splits used:** training used the train splits of BANKING77, CLINC150 and GSM8K, plus WinoGrande train, Amazon ESCI
  train queries and Lichess puzzles (a different dataset from ChessBench's). No test split of any benchmark was used.
- **Overlap found:** an audit of the training text against the public suite found 19 test items that are exact or near-exact
  duplicates of train-split utterances. Following the board's contamination rule, they are counted as wrong in the 56.57 above:
  - `4:BANKING77:BANKING77:test:` 554, 677, 727, 742, 747, 751, 754, 1432, 1735, 1754, 1993, 2081, 2211, 2440, 2644, 2651, 2987,
    3070;
  - `5:CLINC150+OOS:CLINC150+OOS:test:1591`.

## Latency (our approximation)

These numbers come from 1 x RTX PRO 6000, one request at a time, measured through the kit's runner. The sample is our own 760 rows
of the public suite: 750 timed plus 10 warm-up. This approximates the board's check, which uses a private sample.

| median | mean | 80th percentile |
|---|---|---|
| 114 ms | 509 ms | 364 ms |
