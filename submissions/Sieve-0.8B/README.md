# Sieve-0.8B evaluation harness

Sieve-0.8B (`sthanika-ai/Sieve-0.8B` at `31c60b2c9d1f05e32d6a3436dd43241805bdd692`) is the first 16 of 24 blocks of `Qwen/Qwen3.5-0.8B-Base` at `dc7cdfe2`, fully fine-tuned (332.1M parameters; the embeddings are frozen), plus a scalar scorer head: **0.59B parameters run** (332.6M trained). The engine, `sieve_engine:SieveEngine`, ships in the model repo.

## Run

```sh
pip install "torch==2.8.0" "transformers==5.17.0" safetensors huggingface_hub
pip install flash-linear-attention==0.5.2 causal-conv1d==1.7.0   # fast kernels; the run used them
hf download sthanika-ai/Sieve-0.8B --revision 31c60b2c9d1f05e32d6a3436dd43241805bdd692 --local-dir Sieve-0.8B
PYTHONPATH=Sieve-0.8B python -m decision_index run --edition 0.3 --engine sieve_engine:SieveEngine \
    --option sieve_dir=Sieve-0.8B --option max_batch_tokens=49152 --option mem_fraction=0.16 --out runs/Sieve-0.8B
python -m decision_index score --edition 0.3 --results runs/Sieve-0.8B/results.jsonl
```

The base model downloads on first use. The run used per-process caps of 0.12 to 0.25 of an 80 GB card (0.16 is about 12.7 GiB); for the fastest single-request path, drop both options.

## What the engine does

- **Serving:** `modeling_sieve.py` loads `full_weights.safetensors` (blocks 1-16) over the bf16 base model and the head from `head.safetensors`. Scores are divided by the temperature in `config.json` (1.1208), which changes no answer.
- **One question:** the state and the question, with every option listed in an order fixed by a hash of each option's text, are encoded once. The cache is copied once per option and each option continues from its own copy, so no option sees another and reordering the options changes nothing.
- **Several questions:** all of a request's questions are scored together, prefixes grouped by length under a token budget. This changes no answer.
- **`noul`:** answered as a 2-option choice over its own `criteria` descriptions, or plain "No." / "Yes." when none are given.
- **What it never does:** truncate, drop options, tune prompts per benchmark or see gold labels.
- **Declared limits:** state + question + the longest option must fit in 32,768 tokens, otherwise `Unsupported`; no 0.3 request reached it. A question whose options have identical text is refused (they cannot be scored apart): 2 scored requests.
- **Out of memory:** options are scored in halving chunks (exact, since options are independent); if even one option does not fit, it is an ordinary error, so a resumed run retries it. None remained.
