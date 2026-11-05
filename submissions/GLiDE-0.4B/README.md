# GLiDE 0.4B

- Model: [`fastino/GLiDE-0.4B`](https://huggingface.co/fastino/GLiDE-0.4B) @ `6e2671b0ccc8375f2b1c78032dc43be13d82e19b`
- Edition: 0.3, full public suite. `scores.json` has `"complete": true`: 140,178 of 140,178 scored requests answered.
- Decision Index: **51.10** (raw index 63.66)

## Results

- Dataset: [`fastino/GLiDE-0.4B-decision-index`](https://huggingface.co/datasets/fastino/GLiDE-0.4B-decision-index/tree/b10049502b16e0e7ade3fab972c9577998e1a688/runs/glide-0.4b) @ `b100495`
- Contents of `runs/glide-0.4b/`: `results.jsonl.gz`, `scores.json`, `index.json`, `benchmark-summary.json`, `environment.json`, `status.json`. The kit's `upload_run` uploaded them; the results file is unmodified.
- Rescoring with the kit at `25c6ec7` (`score --edition 0.3`) gives 51.10, with all 43 benchmark entries identical.

## How it was run

- Kit: `25c6ec7`, stock `http` engine, no changes to the evaluation code or data.
  - The 150,759 requests of the 0.2.1 suite were run in 16 parts, then the documented 0.3 resume ran the 2,638 rebuilt GSM8K requests.
- Server: the model repo's own `/v1/systemone` server, started from the downloaded repo:

```
GLIDE_MODEL_NAME="GLiDE 0.4B" GLIDE_MAX_BATCH=1 PYTHONPATH=. python -m glide.server --model . --port 8000
python -m decision_index pipeline --engine http --option base_url=http://127.0.0.1:8000 --option "model=GLiDE 0.4B" --out runs/glide-0.4b
```

- Settings: everything the server reads is in the repo. `GLIDE_MAX_BATCH=1`; all other server settings are at their defaults.
- Hardware: NVIDIA RTX PRO 6000 (96 GB), bf16, one server per GPU and one request at a time.
  - Each of the 16 parts ran on its own RTX PRO 6000; the parts were concatenated in order and scored once.

## Declared limits

None were reached: no request was refused or truncated, and no options were removed.
