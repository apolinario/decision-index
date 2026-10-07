# Sieve-2B-Plus evaluation harness

Sieve-2B-Plus (`sthanika-ai/Sieve-2B-Plus` at `ce762346cd0dd6581532b4e9bb19ca1bccd25b8c`) is a LoRA adapter (r 64, α 128, on the attention, MLP and Gated DeltaNet projections) plus a scalar scorer head on `Qwen/Qwen3.5-2B-Base` at `b1485b2f`, truncated to the base model's first 16 of 24 blocks: **1.47B parameters run** (45.9M trained). The engine, `sieve_engine:SieveEngine`, ships in the model repo.

## Run

```sh
pip install "torch==2.8.0" "transformers==5.17.0" "peft==0.21.0" safetensors huggingface_hub
pip install flash-linear-attention==0.5.2 causal-conv1d==1.7.0   # fast kernels; the run used them
hf download sthanika-ai/Sieve-2B-Plus --revision ce762346cd0dd6581532b4e9bb19ca1bccd25b8c --local-dir Sieve-2B-Plus
PYTHONPATH=Sieve-2B-Plus python -m decision_index run --edition 0.3 --engine sieve_engine:SieveEngine \
    --option sieve_dir=Sieve-2B-Plus --option max_batch_tokens=49152 --option mem_fraction=0.16 --out runs/Sieve-2B-Plus
python -m decision_index score --edition 0.3 --results runs/Sieve-2B-Plus/results.jsonl
```

The base model downloads on first use; about 14 GB of GPU memory is needed at peak. `mem_fraction=0.16` caps the process at about 12.7 GiB on an 80 GB card (use about 0.134 for the same cap on a 96 GB card). For the fastest single-request path, drop both options.

## What the engine does

- **Serving:** the adapter and head load onto the bf16 backbone. Scores are divided by the temperature in `config.json` (1.4785), which changes no answer.
- **One question:** the state and the question, with every option listed in an order fixed by a hash of each option's text, are encoded once. The cache is copied once per option and each option continues from its own copy, so no option sees another and reordering the options changes nothing.
- **Several questions:** all of a request's questions are scored together, prefixes grouped by length under a token budget. This changes no answer.
- **`noul`:** answered as a 2-option choice over its own `criteria` descriptions, or plain "No." / "Yes." when none are given.
- **What it never does:** truncate, drop options, tune prompts per benchmark or see gold labels.
- **Declared limits:** state + question + the longest option must fit in 32,768 tokens, otherwise `Unsupported`; no 0.3 request reached it. A question whose options have identical text is refused (they cannot be scored apart): 2 scored requests.
- **Out of memory:** options are scored in halving chunks (exact, since options are independent); if even one option does not fit, it is an ordinary error, so a resumed run retries it. None remained.
