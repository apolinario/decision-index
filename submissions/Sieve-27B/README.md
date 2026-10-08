# Sieve-27B evaluation harness

Sieve-27B (`sthanika-ai/Sieve-27B` at `7836ff5604cb6682c835745fffd1b03a5ce4145c`) is a LoRA adapter plus a pointer head on `Qwen/Qwen3.8-27B`
(`1d4bf0f2`). It reads the state once and scores every supplied option directly; it generates nothing. The engine is
`sieve.decision_index_engine:SieveEngine` from [github.com/sthanika-ai/Sieve](https://github.com/sthanika-ai/Sieve)
at `5284de70197642dbb4d55b4cf05a690b2955b080`.

## Run

```sh
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==5.17.0 peft==0.21.0 huggingface_hub==1.32.0 flash-linear-attention==0.5.2 fla-core==0.5.2 numpy httpx
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128   # if the line above upgraded torch
pip install --no-build-isolation causal-conv1d==1.7.0
pip install --no-deps "sieve-decisions @ git+https://github.com/sthanika-ai/Sieve@5284de70197642dbb4d55b4cf05a690b2955b080"
pip install --no-deps "decision-index @ git+https://github.com/apolinario/decision-index@62d2f51"
hf download sthanika-ai/Sieve-27B --revision 7836ff5604cb6682c835745fffd1b03a5ce4145c --local-dir Sieve-27B
python -m decision_index pipeline --engine sieve.decision_index_engine:SieveEngine --option model=Sieve-27B --out runs/Sieve-27B
```

The backbone is downloaded at its pinned revision on first use, and about 55 GB of GPU memory is needed.

## What the engine does

- **Serving:** the model runs exactly as it is served.
  - The adapter is merged into the bf16 backbone and the pointer head runs in fp32.
  - Probabilities are divided by the temperature stored in `head.pt` (1.050), which changes no answer.
- **One question:** the state and the question run as one causal row.
- **Several questions:** the state is prefilled once and the questions continue from its cache, in batches sized to
  a memory budget. Batching changes no answer.
- **What it never does:** truncate, drop options, tune prompts per benchmark or see gold labels.
- **Declared limits:** a 65,536-token state, a 32,768-token question and up to 255 choice options. Requests beyond them
  are `Unsupported`; no request reached them.
- **Out of memory:** raised as an ordinary error, so a resumed run retries it. None occurred.
- **causal-conv1d:** our run used the compiled kernel (1.7.0). Without it, transformers falls back to a slower
  PyTorch convolution, whose probabilities can differ slightly.
- **Latency:** the engine times the eager path. The package's server (`sieve-serve --graphs`) also has a CUDA-graph
  path, which is faster for short requests.
