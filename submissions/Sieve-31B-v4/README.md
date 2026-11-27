# Sieve-31B-v4 evaluation harness

Sieve-31B-v4 is an open-weights decision model by Sthānika AI, a fine-tune of `google/gemma-4-31B-it` at `842da379` in
two stages: a full fine-tune of every linear projection of its 60 decoder layers (the token embeddings, tied with the
output layer, and every norm are frozen), then a rank-32 LoRA stage merged into the dense weights. It is stored as a
full bf16 model plus `head.pt`, and its weights are public under Apache-2.0:
[`sthanika-ai/Sieve-31B-v4`](https://huggingface.co/sthanika-ai/Sieve-31B-v4). It answers through its own output
layer, from the logits of the option labels at an `Answer: (` slot; it generates nothing.

The engine is `sieve.decision_index_engine:SieveEngine` from the open-source `sieve` package 0.5.2
([github.com/sthanika-ai/Sieve](https://github.com/sthanika-ai/Sieve) at
`aafc31b781e0f590d180d2b1c8ade283d1d8401a`).

## Run

Python 3.12:

```sh
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r https://raw.githubusercontent.com/sthanika-ai/Sieve/aafc31b781e0f590d180d2b1c8ade283d1d8401a/requirements.txt
pip install --no-deps "sieve-decisions @ git+https://github.com/sthanika-ai/Sieve@aafc31b781e0f590d180d2b1c8ade283d1d8401a"
pip install --no-deps "decision-index @ git+https://github.com/apolinario/decision-index@f5a6e4f"
hf download sthanika-ai/Sieve-31B-v4 --revision 464dc7f46e150407aec322634d321071ba4b433f --local-dir Sieve-31B-v4
python -m decision_index run --engine sieve.decision_index_engine:SieveEngine --option model=Sieve-31B-v4 --out runs/Sieve-31B-v4
```

The engine reads everything from `head.pt`; on one GPU no other option is needed. The weights take 58.3 GiB of GPU
memory in bf16. The architecture and size are those of Sieve-31B-v3, whose largest public request (53,703 tokens, 17
questions) peaked at 81.4 GiB allocated, so one RTX PRO 6000 (96 GB) runs every public request. On GPUs with less
memory, `--option devices=cuda:0,cuda:1` (or more GPUs) splits the decoder layers over them.

## The submitted run

The complete public run (`runs/Sieve-31B-v4/` in
[sthanika-ai/Sieve-31B-v4-decision-index-results](https://huggingface.co/datasets/sthanika-ai/Sieve-31B-v4-decision-index-results/tree/8ca746143135e0e1eb390ae58d7c7439a5b3b786))
used the same files (checksums in the model repository's `provenance.json`; `head.pt` sha256 `afb48e59…`) and this
engine, bf16, eager, one request at a time per process. The suite ran in 56 shards on NVIDIA A100-SXM4-40GB GPUs, each
engine process splitting the layers over two or four GPUs with `--option devices=...`; requests that did not fit in
memory on two GPUs were run again on four. The run used the engine code before its release as 0.5.2; it differs only
in comments, the version number and one added argument check. Public index **72.01**; all 140,178 scoreable requests
`ok`, nothing unsupported or truncated.

## Inference settings

All settings are fixed in `head.pt`; the engine needs no option besides `model` (and `devices` to split the model).

- The prompt is Gemma's own chat template with thinking off: a shared block (system turn, the list of questions, the
  state), then one block per question with its labelled options (A, B, …, up to 255) and an `Answer: (` slot.
- The answer is the softmax of the model's output-layer logits of the question's option labels at its slot, divided
  by the calibrated temperature of the question's category (question type, number of options and kind of state), else
  the global one (1.167). This changes no answer.
- Every choice question whose options may be reordered is read twice, with its options as given and rotated by half,
  and the two log-probability vectors are averaged.
- In the 10 global layers the shared block attends only to itself and each question's block attends to the shared
  block and to itself, in both directions; the 50 sliding-window layers are causal. Questions never see each other.
  The model was trained with this attention.
- With a shared block of 2,048 tokens or more, the shared block is computed once and its keys and values serve every
  question and both option orders; this gives the same result as computing it with each question.
- Nothing is truncated, no option is dropped, and no prompt is changed per benchmark.

## Limits

Up to 255 choice options per question; a request beyond that is `Unsupported`, and none was. No token limit: the model
was trained on prompts of up to 16,384 tokens, and every public request (up to about 54,000 tokens) was answered.
Running out of GPU memory raises an ordinary error, which a resumed run retries.

## Latency

Our run split the model over two or four A100-40GB GPUs, so its timings (the kit's `model_request_wall_ms`: median
338 ms, mean 1,603 ms, 95th percentile 2,192 ms) are not comparable with one RTX PRO 6000. Please measure latency on
your RTX PRO 6000 sample, on one GPU with no `devices` option.
