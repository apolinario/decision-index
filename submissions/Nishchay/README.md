# Nishchay

- Model: [`DhruShah1001/Nishchay-1B`](https://huggingface.co/DhruShah1001/Nishchay-1B/tree/217fb8b31b53a69388bafe5bd89e5ebdd0aa8148) @ `217fb8b`
  (gated, access on request; a 1B Llama-architecture decoder with per-type decision heads and its own `/v1/systemone`
  server, `serve.py`). Research preview, shared for evaluation on the Decision Index.
- Edition: 0.3, full public suite. `scores.json` has `"complete": true`: 140,171 of 140,178 scored requests answered,
  7 unsupported.
- Decision Index: **52.42** (raw index 64.86, breadth 49.99)

## Results

- Dataset: [`DhruShah1001/nishchay-1b-decision-index`](https://huggingface.co/datasets/DhruShah1001/nishchay-1b-decision-index/tree/0db6789eee14aa4cdaf73a373ec26cb280581898/runs/nishchay-1b-v2) @ `0db6789` (public)
- Contents of `runs/nishchay-1b-v2/`: `results.jsonl.gz` (all 140,620 rows, compact: no `payload` or `raw_output`),
  `scores.json`, `index.json`, `benchmark-summary.json`, and `shards/shard0/environment.json` and `status.json`.
  The results file is unmodified.
- Rescoring with the kit at `25c6ec7` (`score --edition 0.3`) gives 52.42.

## How it was run

- Kit: `25c6ec7`, stock `http` engine and the kit's own runner (`decision_index.runner.run`), no changes to the
  evaluation code or data. One runner process (`code/run_shard.py` in the dataset, with one shard).
- Server: the model repo's own `/v1/systemone` server, started from the repo at `217fb8b`:

```
pip install "torch>=2.8" "transformers>=5.17" safetensors fastapi uvicorn
python serve.py --model_dir . --port 8000          # --max_len 32768 (default)
python -m decision_index pipeline --engine http --option base_url=http://127.0.0.1:8000 --option model=nishchay-1b --out runs/nishchay-1b-v2
```

- Hardware: 1 × NVIDIA H100 80GB, bf16, one request at a time. Median latency 17.1 ms, mean 22.7 ms, p95 72.0 ms per
  request (kit timing, HTTP included).

## Declared limits

- Context window 32,768 tokens. A request whose prompt does not fit is refused with HTTP 422 ("... longer than the
  maximum model length ...") and recorded by the kit as unsupported: 7 ToolRet requests (prompts of ~62k tokens).
  Nothing is truncated, no option is removed, and the prompt template is fixed.
