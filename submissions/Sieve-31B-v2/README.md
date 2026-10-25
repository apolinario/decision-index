# Sieve-31B-v2 evaluation harness

Sieve-31B-v2 (`sthanika-ai/Sieve-31B-v2` at `2c0d64310ac944b350a9e19f1f54dcd9a5bed3eb`) is a full fine-tune of `google/gemma-4-31B-it` at `842da37`
(its 60 text decoder layers, 29.29B parameters; the token embeddings, final norm and output layer are the backbone's).
The trained layers are stored in full as a PEFT adapter (`modules_to_save`). It uses **bidirectional attention** in
its 10 global attention layers (they attend over the whole prompt in both directions); the 50 sliding-window layers stay
causal, as in the base model. It reads the prompt once and returns a probability for every supplied option; it generates
nothing and has no thinking mode. The engine is `sieve.decision_index_engine:SieveEngine` from
[github.com/sthanika-ai/Sieve](https://github.com/sthanika-ai/Sieve) at `3ff9195c902ee1882e3e4bcfcb5bc733b8fcbb7f`.

## Run

```sh
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==5.17.0 peft==0.21.0 accelerate==1.15.0 huggingface_hub==1.32.0 safetensors "numpy<2" jinja2 httpx pydantic
pip install --no-deps decider-ai==1.6.0
pip install --no-deps "sieve-decisions @ git+https://github.com/sthanika-ai/Sieve@3ff9195c902ee1882e3e4bcfcb5bc733b8fcbb7f"
pip install --no-deps "decision-index @ git+https://github.com/apolinario/decision-index@62d2f51"
hf download sthanika-ai/Sieve-31B-v2 --revision 2c0d64310ac944b350a9e19f1f54dcd9a5bed3eb --local-dir Sieve-31B-v2
python -m decision_index pipeline --engine sieve.decision_index_engine:SieveEngine --option model=Sieve-31B-v2 --out runs/Sieve-31B-v2
```

- **Defaults:** besides `model`, the engine needs no options. It loads the backbone at its pinned revision, copies the
  fine-tuned layers in place and reads the attention mode from the release's `training_config.json`.
- **Memory:** the GPU peaked at 86,348 MiB in our check, including PyTorch's cache, so it fits one 96 GB RTX PRO 6000.
- **Smaller GPUs:** `--option split='{"0": "40GiB", "1": "40GiB"}'` spreads the model over two GPUs.

## What the engine does

- **Prompt:** decider-ai 1.6.0's chat layout, used unmodified. Each question is one user turn in Gemma's chat template,
  with thinking off, and has its own `Answer: (` slot. Questions are independent rows over the full state, and the
  options keep the request's order.
- **Score:** the final hidden state at the slot, multiplied by the output-layer rows of the option labels. A softmax
  over the question's options at the fixed temperature 8.5495 gives the probabilities.
- **Bidirectional attention:** the 10 global layers attend to every prompt token, before and after it, so the state is
  read in the light of the question and options that follow it (no mask for a single unpadded row, a key-padding mask
  for a padded batch); the 50 sliding-window layers are causal. This is how the model was trained. The engine reads the
  mode from the release's `training_config.json` (`"attention": "bidir_global"`); a release without it (Sieve-31B) runs
  causal.
- **Never:** it never truncates, drops options, tunes prompts per benchmark or sees gold labels.
- **Declared limits:** up to 255 options per question and 32,768 tokens per row (state plus one question). Larger
  requests are `Unsupported`; no public request reached them.
- **Out of memory:** the forward chunk is halved and the request retried. A single row that still does not fit raises
  an ordinary error, so a resumed run retries it.

## How the submitted run was made

The run used 2× A100 80GB. Requests were answered by our batched evaluator, which runs the same prompts, readout,
bidirectional global attention and weights as the engine but packs many question-rows into one forward pass. It stored probabilities at
T = 1, and they were re-read at T = 8.5495 (softmax(log p / T)) before kit scoring. Two long requests were re-run with the
model split over both GPUs. Code and details are in the results dataset (`runs/Sieve-31B-v2/RUN_NOTES.md`, `code/`).

**Parity.** The engine above was run through the kit's runner, one request at a time, and compared with the submitted
results:

| Hardware | Sample | Answer flips | Notes |
|---|---|---|---|
| 1× RTX PRO 6000 | 400 random public rows | 3 of 876 questions (0.34%) | two near-ties; on one NLI4CT question the submitted run preferred one option by 0.116, the engine the other by 0.066; \|Δp\| median 0.0001, p99 0.039 |

## Latency

Measured with this engine through the kit's runner on 1× RTX PRO 6000 Blackwell Server Edition: single process, eager,
one request at a time, 400 random public 0.3 rows. Please re-measure on your sample.

| Measure | ms |
|---|---|
| Median | **110** |
| Mean | **437** |
| 80th percentile | **462** |
| 95th percentile | 1,763 |
