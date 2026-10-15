# Sieve-2B-v3 evaluation harness

Sieve-2B-v3 (`sthanika-ai/Sieve-2B-v3` at `905613c41f81fe4cc249d42c5f6b06d1e1c2337f`) is a full fine-tune of
`Qwen/Qwen3.5-2B` at `15852e8c` (its 24 decoder layers, 1.37B trained parameters; the token embeddings and the final
norm are frozen) plus a pointer head. The trained layers are stored in full as a PEFT adapter (`modules_to_save`). It
reads the state once and scores every supplied option directly; it generates nothing. The engine is
`sieve.decision_index_engine:SieveEngine` from [github.com/sthanika-ai/Sieve](https://github.com/sthanika-ai/Sieve) at
`943f833136e69a4321cfdd6bfd7b4f338fc17840`.

## Run

```sh
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==5.17.0 peft==0.21.0 huggingface_hub==1.32.0 safetensors pydantic numpy httpx flash-linear-attention==0.5.2 fla-core==0.5.2
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128   # if the line above upgraded torch
pip install wheel packaging ninja
pip install --no-build-isolation causal-conv1d==1.7.0
pip install --no-deps "sieve-decisions @ git+https://github.com/sthanika-ai/Sieve@943f833136e69a4321cfdd6bfd7b4f338fc17840"
pip install --no-deps "decision-index @ git+https://github.com/apolinario/decision-index@f5a6e4f"
hf download sthanika-ai/Sieve-2B-v3 --revision 905613c41f81fe4cc249d42c5f6b06d1e1c2337f --local-dir Sieve-2B-v3
python -m decision_index pipeline --engine sieve.decision_index_engine:SieveEngine --option model=Sieve-2B-v3 --out runs/Sieve-2B-v3
```

The backbone is downloaded at its pinned revision on first use. Our eager server held about 11 GB of GPU memory,
including PyTorch's cache.

## What the engine does

- **Serving:** the model runs exactly as it is served.
  - The fine-tuned layers are merged into the bf16 backbone and the pointer head runs in fp32.
  - Probabilities are divided by the calibrated temperatures stored in `head.pt`: one per question category (question
    type, number of options and kind of state), else the global one (0.968). This changes no answer.
- **One question:** the state and the question run as one causal row.
- **Several questions:** the state is prefilled once and the questions continue from its cache, in batches sized to
  a memory budget. Batching changes no answer.
- **What it never does:** truncate, drop options, tune prompts per benchmark or see gold labels.
- **Declared limits:** a 65,536-token state, a 32,768-token question with its options and up to 255 choice options per
  question. Requests beyond them are `Unsupported`; no request reached them.
- **Out of memory:** raised as an ordinary error, so a resumed run retries it. None occurred.
- **Latency:** the engine times the eager path. The package's server (`sieve-serve --graphs`) also has a CUDA-graph
  path, which is faster for short requests.
