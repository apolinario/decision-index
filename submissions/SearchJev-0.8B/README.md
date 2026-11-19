# SearchJev-0.8B

- Model: [`SearchJev/SearchJev-0.8B-v2`](https://huggingface.co/SearchJev/SearchJev-0.8B-v2) @ `a0200c0044cc562437ba8172929261093864425d`
  (public; full weights fine-tuned from [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B), plus calibration
  files and the `/v1/systemone` server). Licence of the weights and code: CC BY-NC 4.0.
- Edition: 0.3, full public suite. `scores.json` has `"complete": true`: 140,178 of 140,178 scored requests answered.
- Decision Index: **52.74** (raw index 64.84, breadth 50.26)

## Results

- Dataset: [`SearchJev/SearchJev-0.8B-v2-decision-index`](https://huggingface.co/datasets/SearchJev/SearchJev-0.8B-v2-decision-index/tree/7eea5584f5d532320f27100839c0a6a0f56fd61b/runs/searchjev-0.8b) @ `7eea558`
- Contents of `runs/searchjev-0.8b/`: `results.jsonl.gz`, `scores.json`, `index.json`, `benchmark-summary.json`,
  `environment.json`, `status.json`, uploaded with the kit's `upload_run`; the results file is unmodified.
- Access is gated (manual approval) because the results file keeps every request's payload, i.e. text of suite sources
  whose terms do not allow republishing (the run was not `--compact`). **The maintainers' Hugging Face accounts
  `apolinario` and `multimodalart` already have read access; no access request is needed.** We can also share it
  another way if you prefer.
- Rescoring with the kit at `25c6ec7` (`score --edition 0.3`) gives 52.74, with all 43 benchmark entries identical.

## How it was run

- Kit: `25c6ec7`, stock `http` engine, no changes to the evaluation code or data; one uninterrupted `pipeline` run.
- Server: the model repo's own `/v1/systemone` server, started from the downloaded repo:

```
pip install -r requirements.txt
python -m searchjev.server --model . --port 8000
python -m decision_index pipeline --engine http --option base_url=http://127.0.0.1:8000 --option model=searchjev-0.8b --out runs/searchjev-0.8b
```

- Settings: all server defaults (written to `server_settings.json` in the model repo): prompt layout `plain`;
  bidirectional attention in the full-attention layers (the model is trained and served this way; implemented in
  `searchjev/training/arch.py`, no parameters added); the user turn is written twice when the prompt has at most 3,000
  tokens; per-question-type temperatures from `calibration.json`; no truncation.
- Hardware: NVIDIA H200 (141 GB), bf16, one server, one request at a time. The GPU was shared with a training job
  during the run, so the latency recorded by the kit (median 90 ms, mean 142 ms, 95th percentile 294 ms) is an upper
  bound. Our own check on 300 public sample requests, on the same shared GPU: median 58 ms, mean 179 ms, 80th
  percentile 216 ms.

## Declared limits

The prompt budget is 65,536 tokens per prompt: with bidirectional attention, memory grows with the square of the prompt
length (about 23 GB at 65,000 tokens). A request with a longer prompt gets HTTP 422 with the kit's "maximum context
length" marker. The limit was not reached: no request was refused or truncated (the longest prompt of the public suite
has about 23,000 tokens), and no options were removed.

## Training data (summary; details in the model card)

Distilled from our own [SearchJev-27B](https://huggingface.co/SearchJev/SearchJev-27B): three students were trained from
the base model on 2.61 million questions of the teacher's training data, with the teacher's option probabilities as
targets, and merged in weight space. The data is public data only and is the teacher's: our SearchJev unified data,
about 120 public datasets, the train/dev splits of the public sources of Decision Index benchmarks in request format
(declared; deduplicated against the public suite), 33 public datasets from new domains and program-generated items with
exactly computed answers; about 3% of the rows (from the SearchJev unified data) carry labels produced or filtered by
language models. No outputs of other entrants were used.
