# Sieve-0.8B evaluation harness

Sieve-0.8B (`sthanika-ai/Sieve-0.8B` at `8b882ca830d614ce8e2632d650153c7aa9c9e4c2`) is the first 16 of 24 blocks of `Qwen/Qwen3.5-0.8B-Base` at `dc7cdfe2`, fully fine-tuned (332.1M parameters; the embeddings are frozen), plus a scalar scorer head: **0.59B parameters run** (332.6M trained). The engine is `sieve.decision_index_engine:SieveEngine` from [github.com/sthanika-ai/Sieve](https://github.com/sthanika-ai/Sieve) at `304abb2`: the engine the run used (`sieve_engine:SieveEngine`, then in the model repo), moved there unchanged. It gives the same answers.

## Run

```sh
pip install "torch==2.8.0" "transformers==5.17.0" safetensors huggingface_hub
pip install flash-linear-attention==0.5.2 causal-conv1d==1.7.0   # fast kernels; the run used them
pip install "sieve-decisions @ git+https://github.com/sthanika-ai/Sieve@304abb2c5c5c22bffda0bf7c0d39d1d8aa59af01"
hf download sthanika-ai/Sieve-0.8B --revision 8b882ca830d614ce8e2632d650153c7aa9c9e4c2 --local-dir Sieve-0.8B
python -m decision_index run --edition 0.3 --engine sieve.decision_index_engine:SieveEngine \
    --option model=Sieve-0.8B --option max_batch_tokens=49152 --option mem_fraction=0.16 --out runs/Sieve-0.8B
python -m decision_index score --edition 0.3 --results runs/Sieve-0.8B/results.jsonl
```

The base model downloads on first use. The run used per-process caps of 0.12 to 0.25 of an 80 GB card (0.16 is about 12.7 GiB); for the fastest single-request path, drop both options.

## What the engine does

- **Serving:** the engine loads `full_weights.safetensors` (blocks 1-16) over the bf16 base model and the head from `head.safetensors`. Scores are divided by the temperature in `config.json` (1.1208), which changes no answer.
- **One question:** the state and the question, with every option listed in an order fixed by a hash of each option's text, are encoded once. The cache is copied once per option and each option continues from its own copy, so no option sees another and reordering the options changes nothing.
- **Several questions:** all of a request's questions are scored together, prefixes grouped by length under a token budget. This changes no answer.
- **`noul`:** answered as a 2-option choice over its own `criteria` descriptions, or plain "No." / "Yes." when none are given.
- **What it never does:** truncate, drop options, tune prompts per benchmark or see gold labels.
- **Declared limits:** state + question + the longest option must fit in 32,768 tokens, otherwise `Unsupported`; no 0.3 request reached it. A question whose options have identical text is refused (they cannot be scored apart): 2 scored requests.
- **Out of memory:** options are scored in halving chunks (exact, since options are independent); if even one option does not fit, it is an ordinary error, so a resumed run retries it. None remained.
