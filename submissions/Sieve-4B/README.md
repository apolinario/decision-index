# Sieve-4B evaluation harness

Sieve-4B (`sthanika-ai/Sieve-4B` at `fdad222de6c018a8c3613abc623c9005d76657c7`) is a LoRA adapter (rank 32,
every linear layer including the Gated DeltaNet projections) plus a 255-way answer head on `Qwen/Qwen3.5-4B`
(`851bf6e8`). It reads the state once and scores every option of every question directly; it generates nothing and has
no thinking mode. The inference code ships in the model repo (`sieve4b/`) as a `/v1/systemone` server, run here with
the kit's `http` engine.

## Run

```sh
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
pip install transformers==5.17.0 peft==0.21.0 flash-linear-attention==0.5.2 fla-core==0.5.2 safetensors==0.8.0 \
    pydantic==2.13.5 fastapi==0.141.1 uvicorn==0.54.0 huggingface_hub==1.33.0
pip uninstall -y torchvision torchaudio   # only if a preinstalled build doesn't match torch 2.10 (transformers imports it)
hf download sthanika-ai/Sieve-4B --revision fdad222de6c018a8c3613abc623c9005d76657c7 --local-dir Sieve-4B
cd Sieve-4B
python -m sieve4b.server --ckpt . --device cuda:0 --port 8123 --name sieve-4b --prefix-cache-gb 1 --max-batch-tokens 65536 &
pip install "decision-index @ git+https://github.com/apolinario/decision-index@9eb2dbe"
python -m decision_index pipeline --engine http --option base_url=http://127.0.0.1:8123 --option model=sieve-4b --out runs/Sieve-4B
```

`Qwen/Qwen3.5-4B` is downloaded at its pinned revision on first use; about 10 GB of GPU memory is needed plus the state
cache.

## What the server does

- **Serving:** the adapter is merged into the bf16 backbone; the answer head runs in fp32.
- **One state, isolated questions:** the state is prefilled once and each question continues from that cache as its own
  branch, so questions never see each other.
- **Readout:** each question lists its options under codes; a 255-way linear head at the question's decision token scores
  the codes in one forward pass.
- **Yes/no questions** are scored in both option orders and the two are averaged.
- **Temperatures:** probabilities are divided by a per-type temperature from `sieve_config.json` (choice 1.161, yes/no
  1.440, score 1.000), the values the submitted run used. They change confidence only, never an answer.
- **What it never does:** truncate, drop options, change prompts per benchmark or see gold labels.
- **Declared limits:** up to 255 choice options and 10 score levels, 32,768 tokens for the state plus the longest
  question, 65,536 tokens per request; beyond them the request is `Unsupported` (HTTP 422). None of the scored 0.3
  requests reached them.
- **causal-conv1d:** not installed in the run (transformers' PyTorch fallback).

How the run was made, checksums and latency: `runs/Sieve-4B/RUN_NOTES.md` in the results dataset.
