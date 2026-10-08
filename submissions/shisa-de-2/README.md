# Shisa DE-2

[shisa-ai/shisa-de-2](https://huggingface.co/shisa-ai/shisa-de-2) is a LoRA (rank 16) fine-tune of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it), merged into BF16 weights (Apache-2.0). It is read through the open-source [shisa-de](https://github.com/shisa-ai/shisa-de) SDK, which renders each question and reads the answer from the model's next-token probabilities.

This submission has two entries for the same weights. They differ in one SDK setting.

| | Shisa DE-2 | Shisa DE-2 (repeat) |
|---|---|---|
| SDK policy | `repeat-think`, the SDK default | `repeat` |
| How each question is asked | Written twice in one prompt. If the top probability is under 0.7 and the question has at most 26 options, the model thinks for up to 1,024 tokens and is read again. | Written twice in one prompt. Nothing is generated. |
| Decision Index 0.3 public index | 59.01 (raw 68.68) | 57.57 (raw 67.65) |
| Requests | 140,178 of 140,178 scoreable, all answered, none unsupported, no errors | the same |
| Request time, our measurement | median 47 ms, mean 770 ms, 80th percentile 332 ms | median 45 ms, mean 169 ms, 80th percentile 144 ms |
| Results | [`runs/shisa-de-2-adaptive`](https://huggingface.co/datasets/shisa-ai/decision-index-0.3-shisa-de-2/tree/7b4db45ee5ce8b5413c323530e6cc095060fac58/runs/shisa-de-2-adaptive) | [`runs/shisa-de-2-repeat`](https://huggingface.co/datasets/shisa-ai/decision-index-0.3-shisa-de-2/tree/7b4db45ee5ce8b5413c323530e6cc095060fac58/runs/shisa-de-2-repeat) |

Request time was measured on NVIDIA RTX PRO 6000 Blackwell cards with one request in flight per server. A request can hold several questions, and each is read separately.

The first entry chooses to think on questions it is unsure of, so it may belong on the planned Reasoning board; the second never generates. We are submitting both so each can go where it fits. If only one entry per set of weights is accepted, please take the first.

## Run

```sh
git clone https://github.com/shisa-ai/shisa-de              # d172d49
git clone https://github.com/apolinario/decision-index      # 62d2f51
cd shisa-de && python -m venv .venv
.venv/bin/pip install -e . -e ../decision-index

# build the suite into ../decision-index/suite-0.3 as the kit's README describes

CUDA_VISIBLE_DEVICES=0 vllm serve shisa-ai/shisa-de-2 --host 127.0.0.1 --port 8026 &    # vLLM 0.30.0, no other options

.venv/bin/python scripts/decision_index_run.py --suite-dir ../decision-index/suite-0.3 \
    --endpoint http://127.0.0.1:8026 --policy repeat-think --out runs/shisa-de-2-adaptive
.venv/bin/python scripts/decision_index_run.py --suite-dir ../decision-index/suite-0.3 \
    --endpoint http://127.0.0.1:8026 --policy repeat --out runs/shisa-de-2-repeat
```

`decision_index_run.py` calls the kit's own `run` and `score` with the engine class `scripts.decision_index_engine:ShisaDE2Engine`. To call the kit directly:

```sh
python -m decision_index run --engine scripts.decision_index_engine:ShisaDE2Engine \
    --model shisa-ai/shisa-de-2 --option tokenizer=shisa-ai/shisa-de-2 \
    --option base_url=http://127.0.0.1:8026 --option max_tokens=262144 \
    --option serving_manifest=manifest.json --option policy=repeat-think \
    --compact --out runs/shisa-de-2-adaptive
```

`manifest.json` is any JSON file describing the server; the engine copies it into `environment.json`.

- **Engine:** [`scripts/decision_index_engine.py`](https://github.com/shisa-ai/shisa-de/blob/d172d49/scripts/decision_index_engine.py) is a kit `Engine` subclass over the SDK's `DecisionModel.decide`. There is no `/v1/systemone` server; the engine talks to vLLM's `/v1/completions` directly.
  - It renders the model's chat template with thinking off and requests the logprobs of the option codes by token id, then takes a softmax over the listed options. Temperature 1, calibration off.
  - Each question in a request is read separately, one after another.
  - For the thinking read, the model generates up to 1,024 tokens at temperature 0, the client closes the thought, and the option codes are read after it.
- **Fastest path:** `--option policy=repeat` is the fastest path that keeps the repeated question. The SDK contract is in [docs/READOUT-DE2.md](https://github.com/shisa-ai/shisa-de/blob/d172d49/docs/READOUT-DE2.md).
- **Attention kernel:** on SM90 GPUs (H100, H200, H20), vLLM 0.30.0 runs this base model through a FlashAttention 4 path that lowers accuracy. On the RTX PRO 6000 it selects Triton attention and no option is needed. The model card has the details.
- **How the runs were made:** fresh 0.3 runs with the kit at `62d2f51`, no rows carried over. The 140,620 rows were split round-robin into two shards, each run sequentially against its own vLLM server (one RTX PRO 6000 Blackwell each), then concatenated and scored with `score --edition 0.3`.
- **Declared limits:** 256 options per question and a 262,144-token context. A request over either is reported as unsupported, with nothing truncated. No request in the suite reached either.
- **Training data:** described on the [model card](https://huggingface.co/shisa-ai/shisa-de-2#training). DE-2’s training mixture includes examples derived from the upstream BANKING77 training split (`kev-banking-largek`). We have done an exact normalized-state matching and a wrapper-aware BANKING77 utterance audit, although these checks do not rule out paraphrase, translation, or pretrained-backbone contamination.
