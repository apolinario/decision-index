# Sieve-31B evaluation harness

Sieve-31B (`sthanika-ai/Sieve-31B` at `29de6dd7e489ee01f05e35590494adf9400685a2`) is a LoRA adapter (rank 128) on `google/gemma-4-31B-it` (`842da37`).
It reads the prompt once and returns a probability for every supplied option. It generates nothing and has no thinking
mode. The engine is `sieve31b.engine:Sieve31BEngine`, which is part of the model repo.

## Run

```sh
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==5.17.0 peft==0.21.0 accelerate==1.15.0 huggingface_hub==1.32.0 "numpy<2" jinja2 httpx
pip install --no-deps decider-ai==1.6.0
pip install --no-deps "decision-index @ git+https://github.com/apolinario/decision-index@62d2f51"
hf download sthanika-ai/Sieve-31B --revision 29de6dd7e489ee01f05e35590494adf9400685a2 --local-dir Sieve-31B
PYTHONPATH=Sieve-31B python -m decision_index pipeline --engine sieve31b.engine:Sieve31BEngine --out runs/Sieve-31B
```

- **Defaults:** the engine needs no options. It loads the base model at its pinned revision and the adapter from the
  downloaded folder.
- **Memory:** the GPU held about 83 GB during our run, including PyTorch's cache, so it fits one 96 GB RTX PRO 6000.
- **Smaller GPUs:** `--option split='{"0": "40GiB", "1": "40GiB"}'` spreads the model over two GPUs.

## What the engine does

- **Prompt:** decider-ai 1.6.0's chat layout, used unmodified. Each question is one user turn in Gemma's chat template,
  with thinking off, and has its own `Answer: (` slot. Questions are independent rows over the full state, and the
  options keep the request's order.
- **Score:** the final hidden state at the slot, multiplied by the output-layer rows of the option labels. A softmax
  over the question's options at the fixed temperature 8.5495 gives the probabilities. The LoRA adapter is applied
  unmerged with PEFT, in bf16, eager.
- **Never:** it never truncates, drops options, tunes prompts per benchmark or sees gold labels.
- **Declared limits:** up to 255 options per question and 32,768 tokens per row (state plus one question). Larger
  requests are `Unsupported`; no public request reached them.
- **Out of memory:** the forward chunk is halved and the request retried. A single row that still does not fit raises
  an ordinary error, so a resumed run retries it.

## How the submitted run was made

The run used 2× A100 80GB. Requests were answered by our batched evaluator, which runs the same prompts, readout and
adapter as the engine but packs many question-rows into one forward pass. It stored probabilities at T = 1, and they
were re-read at T = 8.5495 (softmax(log p / T)) before kit scoring. Code and details are in the results dataset
(`runs/Sieve-31B/RUN_NOTES.md`, `code/`).

**Parity.** The engine above was run through the kit's runner, one request at a time, and compared with the submitted
results:

| Hardware | Sample | Answer flips | Notes |
|---|---|---|---|
| 1× RTX PRO 6000 | 400 random public rows | 1 of 876 questions (0.11%) | the one flip is a tie; max \|Δp\| 0.057 |
| 2× A100 | 344 rows across all datasets | 7 of 1,783 questions (0.39%) | every flip a near-tie (top-2 margin ≤ 0.034), max \|Δp\| 0.055 |

## Latency

Measured with this engine through the kit's runner on 1× RTX PRO 6000 Blackwell Server Edition: single process, eager,
one request at a time, 400 random public 0.3 rows (prompt lengths match the suite). Please re-measure on your sample.

| Measure | ms |
|---|---|
| Median | 163 |
| Mean | 644 |
| 80th percentile | 675 |
