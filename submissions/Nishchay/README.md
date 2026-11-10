# Nishchay

- Model: [`DhruShah1001/Nishchay-1B`](https://huggingface.co/DhruShah1001/Nishchay-1B) @ `92a8f15`
  (gated, access on request; a 1B Llama-architecture decoder with per-type decision heads and its own `/v1/systemone`
  server, `serve.py`). Research preview, shared for evaluation on the Decision Index.
- Edition: 0.3, full public suite. `scores.json` has `"complete": true`: 140,171 of 140,178 scored requests answered,
  7 unsupported.
- Decision Index: **51.85** (raw index 64.44)

## Results

- Dataset: [`DhruShah1001/nishchay-1b-decision-index`](https://huggingface.co/datasets/DhruShah1001/nishchay-1b-decision-index/tree/e7e7d3b3df7953a442251ab34a5dd7836505321c/runs/nishchay-1b) @ `e7e7d3b`
- Contents of `runs/nishchay-1b/`: `results.jsonl.gz`, `scores.json`, `index.json`, `benchmark-summary.json`, and
  `shards/shard0..7/environment.json` and `status.json` (one per runner process); the results file is unmodified.
- Access is gated (manual approval) because the results keep each request's payload, i.e. text of suite sources whose
  terms do not allow republishing. Access requests from the maintainers are approved.
- Rescoring with the kit at `25c6ec7` (`score --edition 0.3`) gives 51.85.

## How it was run

- Kit: `25c6ec7`, stock `http` engine and the kit's own runner (`decision_index.runner.run`), no changes to the
  evaluation code or data. The suite was split into 8 disjoint shards by `sha256(run_id) % 8` with a small wrapper
  (`run_shard.py`, included in the dataset under `code/`), one runner process per GPU, and the shard results merged
  line by line before scoring.
- Server: the model repo's own `/v1/systemone` server, one replica per GPU, started from the downloaded repo:

```
pip install "torch>=2.8" "transformers>=5.17" safetensors fastapi uvicorn
python serve.py --model_dir . --port 8000          # --max_len 32768 (default)
python -m decision_index pipeline --engine http --option base_url=http://127.0.0.1:8000 --option model=nishchay-1b --out runs/nishchay-1b
```

- Hardware: 8 × NVIDIA H100 80GB, bf16, one request at a time per replica. Median latency 16.7 ms, p95 78.3 ms per
  request (kit timing, HTTP included).

## Declared limits

- Context window 32,768 tokens. A request whose prompt does not fit is refused with HTTP 422 ("... longer than the
  maximum model length ...") and recorded by the kit as unsupported: 7 ToolRet requests (prompts of ~62k tokens).
  Nothing is truncated, no option is removed, and the prompt template is fixed.
