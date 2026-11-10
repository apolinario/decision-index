# SearchJev

- Model: [`SearchJev/SearchJev-27B`](https://huggingface.co/SearchJev/SearchJev-27B) @ `5a81b572d36826e0d1894f94637a56fd279a664f`
  (public; a LoRA adapter on [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), merged into the weights at load
  time, plus calibration files and the `/v1/systemone` server). Licence of the adapter and code: CC BY-NC 4.0.
- Edition: 0.3, full public suite. `scores.json` has `"complete": true`: 140,178 of 140,178 scored requests answered.
- Decision Index: **73.52** (raw index 79.71, breadth 72.97)

## Results

- Dataset: [`SearchJev/SearchJev-27B-decision-index`](https://huggingface.co/datasets/SearchJev/SearchJev-27B-decision-index/tree/9e7b3b0019cb08fa8722eed9f2bfd8706cba03dd/runs/searchjev-27b) @ `9e7b3b0`
- Contents of `runs/searchjev-27b/`: `results.jsonl.gz`, `scores.json`, `index.json`, `benchmark-summary.json`,
  `environment.json`, `status.json`, uploaded with the kit's `upload_run`; the results file is unmodified.
- Access is gated (manual approval) because the results file keeps every request's payload, i.e. text of suite sources
  whose terms do not allow republishing (the run was not `--compact`). Access requests from the maintainers are approved;
  we can also grant access another way if you prefer.
- Rescoring with the kit at `25c6ec7` (`score --edition 0.3`) gives 73.52, with all 43 benchmark entries identical.

## How it was run

- Kit: `25c6ec7`, stock `http` engine, no changes to the evaluation code or data; one uninterrupted `pipeline` run.
- Server: the model repo's own `/v1/systemone` server, started from the downloaded repo:

```
pip install -r requirements.txt
python -m searchjev.server --model . --port 8000 --served-name searchjev-27b
python -m decision_index pipeline --engine http --option base_url=http://127.0.0.1:8000 --option model=searchjev-27b --out runs/searchjev-27b
```

- Settings: all server defaults (written to `server_settings.json` in the model repo): prompt layout `plain`, no
  read-twice, LoRA scale 1.0 merged into the weights, per-question-type temperatures from `calibration.json`, prompt
  budget = the model's context (262,144 tokens), no truncation (a request that does not fit gets HTTP 422 with the kit's
  "maximum context length" marker; none did).
- Hardware: NVIDIA H200 (141 GB), bf16, one server, one request at a time. Our own latency check on the H200 (300 public
  sample requests, one at a time): median 94 ms, mean 312 ms, 80th percentile 413 ms.

## Declared limits

None were reached: no request was refused or truncated, and no options were removed.

## Training data (summary; details in the model card)

Public data only: our SearchJev unified data, about 120 public datasets, the train/dev splits of the public sources of
Decision Index benchmarks rewritten into the request format (declared; deduplicated against the public suite), 33 public
datasets from new domains, and program-generated items with exactly computed answers. About 3% of the training rows (from
the SearchJev unified data) carry labels produced or filtered by language models. No outputs of other entrants were used.
