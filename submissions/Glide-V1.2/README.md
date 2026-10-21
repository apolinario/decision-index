# Glide V1.2

- Model: [`fastino/Glide-V1.2-19.10.26`](https://huggingface.co/fastino/Glide-V1.2-19.10.26) @ `19a948f812af30c8962337fbbec2472e575eee7d` (private; access for the maintainers)
- Edition: 0.3, full public suite. `scores.json` has `"complete": true`: 140,178 of 140,178 scored requests answered.
- Decision Index: **71.99** (raw index 78.89)

## Results

- Dataset: [`fastino/Glide-V1.2-decision-index`](https://huggingface.co/datasets/fastino/Glide-V1.2-decision-index/tree/493da3c076683483d1e8127feb966f5b1c1d2899/runs/glide-v1.2) @ `493da3c`, private
- Contents of `runs/glide-v1.2/`: `results.jsonl.gz`, `scores.json`, `index.json`, `benchmark-summary.json`, `environment.json`, `status.json`. The kit's `upload_run` uploaded them; the results file is unmodified.
- Rescoring with the kit at `25c6ec7` (`score --edition 0.3`) gives 71.99, with all 43 benchmark entries identical.

## How it was run

- Kit: stock `http` engine, no changes to the evaluation code or data.
  - The run used `87d4650` (0.2.1).
  - The documented 0.3 resume at `9eb2dbe` then ran the 2,638 rebuilt GSM8K requests.
- Server: the model repo's own `/v1/systemone` server, started from the downloaded repo:

```
GLIDE_MODEL_NAME="Glide V1.2" GLIDE_MAX_BATCH=1 PYTHONPATH=. python -m glide.server --model . --port 8000
python -m decision_index pipeline --engine http --option base_url=http://127.0.0.1:8000 --option "model=Glide V1.2" --out runs/glide-v1.2
```

- Hardware: NVIDIA RTX PRO 6000 (96 GB), bf16, one server per GPU and one request at a time.
  - The 0.2.1 request set was split into 16 parts, each run on its own RTX PRO 6000.
  - The parts were concatenated in order and scored once.
- Latency during the run (HTTP wall time, ok rows): median 84.4 ms, mean 310.4 ms, p95 1,149.7 ms.

## Declared limits

None were reached: no request was refused or truncated, and no options were removed.
