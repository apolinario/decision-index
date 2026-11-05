# GLiDE 27B

- Model: [`fastino/GLiDE`](https://huggingface.co/fastino/GLiDE) @ `4d32743ae96b74f0116f050d37742bbd43c04e7d`
- The run used `fastino/GLiDE-V1.2.1-Loop` @ `f86708f3708ee546161f2c9b60655226b5bb5d9f` (that repo is now `fastino/GLiDE-27B`). `fastino/GLiDE` @ `4d32743` has the same weights, readout, `decision_config.json` and inference code (identical file hashes); the only code change is the default served model name (`glide-v1.2` to `glide-27b`). The run and its results keep the earlier name.
- Edition: 0.3, full public suite. `scores.json` has `"complete": true`: 140,178 of 140,178 scored requests answered.
- Decision Index: **72.49** (raw index 79.17)

## Results

- Dataset: [`fastino/GLiDE-V1.2.1-Loop-decision-index`](https://huggingface.co/datasets/fastino/GLiDE-V1.2.1-Loop-decision-index/tree/aa491d17b524a3329a93b4392c635a12e15d1a15/runs/glide-v1.2.1-loop) @ `aa491d1`
- Contents of `runs/glide-v1.2.1-loop/`: `results.jsonl.gz`, `scores.json`, `index.json`, `benchmark-summary.json`, `environment.json`, `status.json`. The kit's `upload_run` uploaded them; the results file is unmodified.
- Rescoring with the kit at `25c6ec7` (`score --edition 0.3`) gives 72.49, with all 43 benchmark entries identical.

## How it was run

- Kit: `25c6ec7`, stock `http` engine, no changes to the evaluation code or data.
  - The 150,759 requests of the 0.2.1 suite were run in 16 parts, then the documented 0.3 resume ran the 2,638 rebuilt GSM8K requests.
- Server: the model repo's own `/v1/systemone` server, started from the downloaded repo:

```
GLIDE_MODEL_NAME="GLiDE V1.2.1 Loop" GLIDE_MAX_BATCH=1 PYTHONPATH=. python -m glide.server --model . --port 8000
python -m decision_index pipeline --engine http --option base_url=http://127.0.0.1:8000 --option "model=GLiDE V1.2.1 Loop" --out runs/glide-v1.2.1-loop
```

- Settings: everything the server reads is in the repo. `GLIDE_MAX_BATCH=1`; all other server settings are at their defaults.
- Hardware: NVIDIA RTX PRO 6000 (96 GB), bf16, one server per GPU and one request at a time.
  - Each of the 16 parts ran on its own RTX PRO 6000; the parts were concatenated in order and scored once.

## Declared limits

None were reached: no request was refused or truncated, and no options were removed.
