# Model submissions


## Text

| Model | Edition | Decision Index | Complete results | Declared limits |
|---|---|---:|---|---|
| [Koa-Decisions](https://huggingface.co/SF-Foundation-DS/Koa-Decisions/tree/db5194c0743d0f77f571841a29181a16075b1408) (private) | 0.3 | 75.46 | [scores.json](https://huggingface.co/datasets/SF-Foundation-DS/Koa-Decisions-results/blob/bbd4271c0a6be2dc950012d44ffc344e956b3be2/runs/koa-decisions/scores.json) (private) | 65,536 tokens per forward sequence (`max_length=65536`); a question that does not fit with its context is answered `unsupported`, never truncated |

- Complete results: https://huggingface.co/datasets/SF-Foundation-DS/Koa-Decisions-results/tree/bbd4271c0a6be2dc950012d44ffc344e956b3be2
- Hardware: 8× NVIDIA H200 (one kit process per GPU over interleaved row shards), bf16
- Access: both repos are private in the `SF-Foundation-DS` organisation; see the pull request.


## Run

Download the model at the pinned revision; the engine and the run scripts are in its `submission/` folder. Build the environment with Python 3.12: torch 2.14.0 for CUDA 13.0, then exactly the versions in `submission/requirements.txt`, which include the kit at `25c6ec7`:

```bash
hf download SF-Foundation-DS/Koa-Decisions --revision db5194c0743d0f77f571841a29181a16075b1408 --local-dir Koa-Decisions
bash Koa-Decisions/submission/setup_venv.sh venv
```

Run and score the public suite with the kit, from a directory that contains `suite-0.3/`:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHONPATH=$PWD/Koa-Decisions/submission venv/bin/python -m decision_index run \
  --edition 0.3 --suite-dir suite-0.3 \
  --engine koa_decisions_engine:KoaDecisionsEngine \
  --option checkpoint=$PWD/Koa-Decisions --option max_length=65536 \
  --out runs/koa-decisions
venv/bin/python -m decision_index score --edition 0.3 --suite-dir suite-0.3 \
  --results runs/koa-decisions/results.jsonl --engine koa-decisions --out runs/koa-decisions
```

The scored run used `submission/run_public_suite.sh`. It runs one kit process per GPU over 16 interleaved shards of the scored rows, merges the results back in suite order and scores them with the kit; `submission/README.md` explains it.
