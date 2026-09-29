# Sieve-2B evaluation harness

Sieve-2B (`sthanika-ai/sieve-2b` at `5aa4d83a4096bcb6d0e0ae75738a225a8fad4b1e`) is a LoRA adapter plus a scalar scorer head on
`Qwen/Qwen3.5-2B-Base`, truncated to the base model's first 16 of 24 blocks: **1.43B parameters run** (4.69M
trained). It reads the state and question once and scores every supplied option as an isolated continuation of that
shared prefix; it generates nothing. The engine, `sieve_engine:SieveEngine`, ships in the model repo at the same revision.

## Run

```sh
pip install "torch==2.8.0" "transformers==5.17.0" "peft==0.21.0" safetensors huggingface_hub
pip install flash-linear-attention==0.5.2 causal-conv1d==1.7.0   # optional fast kernels; the run used them
hf download sthanika-ai/sieve-2b --revision 5aa4d83a4096bcb6d0e0ae75738a225a8fad4b1e --local-dir sieve-2b
PYTHONPATH=sieve-2b python -m decision_index run --engine sieve_engine:SieveEngine \
    --option sieve_dir=sieve-2b --option max_batch_tokens=49152 --option mem_fraction=0.16 --out runs/Sieve-2B
python -m decision_index score --results runs/Sieve-2B/results.jsonl
```

The Qwen3.5-2B-Base backbone is downloaded on first use; about 14 GB of GPU memory is needed at peak.
The run used `max_batch_tokens=49152` and `mem_fraction=0.16` (a ~12.7 GiB cap on an 80 GB A100; use ~0.134 for the
same cap on a 96 GB card). On an A100 these reproduce the recorded probabilities bit-for-bit (100 of 100 re-run requests).
The cap made a few long POP909 requests score their options in chunks: without it the answers are identical and those
probabilities differ by up to 1e-3 (98 of 100 bit-identical); with default options the answers are identical and
probabilities differ by up to 3e-2 (bf16 batching). Other GPU architectures can differ in the last digits.

## What the engine does

- **Serving:** the model runs as released. LoRA and the head load onto the bf16 backbone; probabilities are divided by
  the temperature stored in `config.json` (1.7144), which changes no answer.
- **One question:** the state and the question (with every option listed, in an order fixed by a hash of each option's
  own text) are encoded once; the cache is copied once per option and each option continues from its own copy, so no
  option sees another and reordering the options changes nothing.
- **Several questions:** all of a request's questions are scored together, prefixes grouped by length under a token
  budget; this changes no answer.
- **`noul`:** answered as a 2-option choice over its own `criteria` descriptions, or plain "No." / "Yes." when none are given.
- **What it never does:** truncate, drop options, tune prompts per benchmark or see gold labels.
- **Declared limits:** state + question + the longest option must fit in 32,768 tokens, otherwise `Unsupported`; no
  0.2.1 request reached it. A question whose options have identical text is refused (they cannot be scored apart):
  2 scored requests (1 MMLU-Pro, 1 BBH), plus 23 among the official exclusions.
- **Out of memory:** options are scored in halving chunks (exact, since options are independent); if even one option
  does not fit, it is an ordinary error, so a resumed run retries it. None remained.
- **Latency:** single process, one request at a time, bf16, on an otherwise idle A100 80GB: median 85 ms, p95 223 ms
  (400 uniformly sampled suite requests). The `latency_ms` in `scores.json` (median 143 ms) was recorded during the
  shared 7-process run.
