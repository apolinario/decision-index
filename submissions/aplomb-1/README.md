# Aplomb 1

[empiriolabsai/aplomb-1](https://huggingface.co/empiriolabsai/aplomb-1/tree/d67432916c6e1fe569ab2de7942755eee91e6880) is a 5.3B decision model from
EmpirioLabs, built on Qwen3.5-4B, with open weights under the EmpirioLabs Model License (the repository is gated and
access is approved automatically). It answers each question with a probability for every option instead of
generated text.

| | |
|---|---|
| Decision Index 0.3 public index | **43.49** (raw 57.04, breadth skill 42.39) |
| Area skill | knowledge 29.8 · language 47.2 · retrieval 44.5 · tools 61.6 · arts 34.0 |
| Requests | complete: 140,178 of 140,178 scoreable (140,620 sent), all answered, none unsupported, nothing truncated, no errors |
| Results | [empiriolabsai/decision-index-results-aplomb-1](https://huggingface.co/datasets/empiriolabsai/decision-index-results-aplomb-1/tree/490b0a103e0c4febacdb56b6456b12f34e5426d2/runs/aplomb-1-0.3) (`--compact`, no suite text) |
| Latency (serial) | median 39 ms, mean 364 ms, p80 399 ms: one request in flight on one NVIDIA RTX PRO 6000 Blackwell Server Edition, the same server and settings, a stratified 2,000-request sample of the 0.3 suite (`suite sample --n 2000 --seed 20261008`) |

## Running it

```sh
hf download empiriolabsai/aplomb-1 --revision d67432916c6e1fe569ab2de7942755eee91e6880 --local-dir aplomb-1   # gated; access is approved automatically
pip install -U "transformers>=5.17" torch accelerate torchvision torchaudio torchcodec soundfile librosa flash-linear-attention
python aplomb-1/serve_aplomb.py --model aplomb-1 --port 8000      # POST /v1/systemone on 127.0.0.1:8000, GET /health

git clone https://github.com/apolinario/decision-index && cd decision-index && git checkout 9eb2dbe
pip install -e ".[rebuild]"
python -m decision_index suite rebuild --work work
python -m decision_index suite import \
    --rows work/artifacts/benchmark-suite/release-v2-rebuilt/selected-rows.jsonl.gz \
    --added-rows work/artifacts/benchmark-suite/release-v2-rebuilt/added-rows.jsonl.gz \
    --gsm8k-rows work/artifacts/benchmark-suite/release-v3-rebuilt/gsm8k-rows.jsonl.gz
python -m decision_index pipeline --engine http --compact \
    --option base_url=http://127.0.0.1:8000 --option model=aplomb-1 --out runs/aplomb-1
python -m decision_index score --edition 0.3 --results results.jsonl.gz --out rescore   # the published file
```

- **Server:** `serve_aplomb.py` loads the model once and answers one request at a time with the reference inference
  in `run_aplomb.py` and `aplomb/`. Every token of the state is read and nothing is truncated.
- **Settings:** the server's defaults: bf16 weights; every choice is also asked with its options in reverse order
  (`--debias`); the per-type temperatures in `edm_config.json`; 8 questions per forward batch. Requests carry only
  `model`, `state` and `questions`. The weights need about 12 GB of accelerator memory.
- **Software in this run:** the install line above on October 8, 2026: torch 2.14.1+cu130, transformers 5.19.0,
  flash-linear-attention 0.5.2.
- **Images, video and audio (for the Vision board):** the same server reads media objects anywhere in `state`, as
  `{"type": "image", "data": "<base64>", "mime": "image/png"}` or `{"type": "image", "url": "https://..."}`, and
  the same with `"video"` or `"audio"`; the settings above apply unchanged.

## How the run was made

- Kit `9eb2dbe` (main; the same tree as `62d2f51`), the suite rebuilt and imported as above and verified against
  the 0.3 hashes.
- 16 machines, each with one NVIDIA RTX PRO 6000 Blackwell (10 Server Edition, 3 Max-Q Workstation Edition, 3 Workstation Edition), one server per GPU and one request in flight. The suite was split into 16 shards by a hash of `run_id`. Each shard ran on its own machine with the
  kit's own runner and `http` engine, with the edition filter and corpus hash exactly as `pipeline` passes them:

  ```python
  suite = Suite("suite-0.3", "0.3"); corpus = suite.verify(strict=True)["sha256"]
  keep = lambda e: suite.in_edition(e) and int(hashlib.sha256(e["run_id"].encode()).hexdigest(), 16) % 16 == shard
  run("http", {"base_url": "http://127.0.0.1:8000", "model": "aplomb-1"}, suite.row_paths, out, corpus_sha256=corpus, keep=keep)
  ```

- The 16 results files were concatenated, each request once, and scored with `score --edition 0.3`.
  `results.jsonl.gz` is those rows in the `--compact` format (payload and raw output omitted), otherwise unedited;
  re-scoring it gives 43.49 and `"complete": true`. `environment.json` is shard 0's; every shard's own
  `environment.json` and `status.json` are in `shards/`.
- The request times in the results come from those 16 machines, one request in flight on each, so they mix three
  GPU editions (median 58 ms, mean 228 ms over the whole suite); the serial latency above is the
  single-GPU measurement.

## Declared limits

Up to 1,000,000 tokens of state plus question, 2 to 255 options per choice and 512 evaluated questions per request.
No suite request comes near them: 0 unsupported, nothing truncated.

## Training data

Training data included the public train splits of WinoGrande and ContractNLI, two of the suite's benchmarks. We could
not fully verify that suite items were excluded from the earliest training data.
