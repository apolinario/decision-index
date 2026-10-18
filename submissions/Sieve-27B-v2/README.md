# Sieve-27B-v2 evaluation harness

Sieve-27B-v2 (`sthanika-ai/Sieve-27B-v2` at `f84ebfaca4073850add1f37eec4f3474d40093bb`) is a full fine-tune of
`Qwen/Qwen3.8-27B` at `1d4bf0f2` (its 64 decoder layers, 24.35B trained parameters; the token embeddings and the final
norm are frozen) plus a pointer head. The trained layers are stored in full as a PEFT adapter (`modules_to_save`). It
reads the state once and scores every supplied option directly; it generates nothing. The engine is
`sieve.decision_index_engine:SieveEngine` from [github.com/sthanika-ai/Sieve](https://github.com/sthanika-ai/Sieve) at
`5f1e35791c64aaa39c76d4b20b79cc97490e95e2`.

## Run

```sh
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==5.17.0 peft==0.21.0 huggingface_hub==1.32.0 safetensors pydantic numpy httpx flash-linear-attention==0.5.2 fla-core==0.5.2
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128   # if the line above upgraded torch
pip install --no-deps triton==3.7.1
pip install --no-deps "sieve-decisions @ git+https://github.com/sthanika-ai/Sieve@5f1e35791c64aaa39c76d4b20b79cc97490e95e2"
pip install --no-deps "decision-index @ git+https://github.com/apolinario/decision-index@f5a6e4f"
hf download sthanika-ai/Sieve-27B-v2 --revision f84ebfaca4073850add1f37eec4f3474d40093bb --local-dir Sieve-27B-v2
python -m decision_index pipeline --engine sieve.decision_index_engine:SieveEngine --option model=Sieve-27B-v2 --out runs/Sieve-27B-v2
```

The backbone is downloaded at its pinned revision on first use. The model needs about 52 GB of GPU memory in bf16,
so one 80 GB or 96 GB GPU is enough. Our run did not use causal-conv1d (it has no kernels for the B300), so
transformers' PyTorch convolution ran; on a GPU where causal-conv1d is installed, probabilities can differ slightly.

## What the engine does

- **Serving:** the model runs exactly as it is served.
  - The fine-tuned layers are copied into the bf16 backbone in place (`sieve` 0.3.2 and later) and the pointer head
    runs in fp32.
  - Probabilities are divided by the calibrated temperatures stored in `head.pt`: one per question category (question
    type, number of options and kind of state), else the global one (0.999). This changes no answer.
- **One question:** the state and the question run as one causal row.
- **Several questions:** the state is prefilled once and the questions continue from its cache, in batches sized to
  a memory budget. Batching changes no answer.
- **What it never does:** truncate, drop options, tune prompts per benchmark or see gold labels.
- **Declared limits:** a 65,536-token state, a 32,768-token question with its options and up to 255 choice options per
  question. Requests beyond them are `Unsupported`; no request reached them.
- **Out of memory:** raised as an ordinary error, so a resumed run retries it. None occurred.
- **Latency:** the engine times the eager path. The package's server (`sieve-serve --graphs`) also has a CUDA-graph
  path, which is faster for short requests.
