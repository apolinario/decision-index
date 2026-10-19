# Shisa DE-2.1

[shisa-ai/shisa-de-2.1](https://huggingface.co/shisa-ai/shisa-de-2.1) is [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) with the mean of eleven LoRA (rank 16) fine-tunes, trained with different seeds, merged into BF16 weights (Apache-2.0). It is the successor to [Shisa DE-2](https://huggingface.co/shisa-ai/shisa-de-2), with the same prompt, readout and policy, and is read through the open-source [shisa-de](https://github.com/shisa-ai/shisa-de) SDK, which renders each question and reads the answer from the model's next-token probabilities.

This submission has two entries for the same weights. They differ in one SDK setting.

| | Shisa DE-2.1 | Shisa DE-2.1 (repeat) |
|---|---|---|
| SDK policy | `repeat-think`, the SDK default | `repeat` |
| How each question is asked | Written twice in one prompt. If the top probability is under 0.7 and the question has at most 26 options, the model thinks for up to 1,024 tokens and is read again. | Written twice in one prompt. Nothing is generated. |
| Decision Index 0.3 public index | 60.89 (raw 70.18) | 59.23 (raw 69.01) |
| Requests | 140,178 of 140,178 scoreable, all answered, none unsupported, no errors | the same |
| Request time, our measurement | median 55 ms, mean 1,197 ms, 80th percentile 417 ms | median 43 ms, mean 163 ms, 80th percentile 143 ms |
| Results | [`runs/shisa-de-2.1-adaptive`](https://huggingface.co/datasets/shisa-ai/decision-index-0.3-shisa-de-2.1/tree/f25949b7a9b88592a00507453b42272ecf3def99/runs/shisa-de-2.1-adaptive) | [`runs/shisa-de-2.1-repeat`](https://huggingface.co/datasets/shisa-ai/decision-index-0.3-shisa-de-2.1/tree/f25949b7a9b88592a00507453b42272ecf3def99/runs/shisa-de-2.1-repeat) |

Request time was measured one request at a time on one NVIDIA RTX PRO 6000 Blackwell with vLLM default settings, over 700 requests of the board's sample mix (the latency-v1 rows per benchmark, spread over payload length) after 10 warmup requests. A request can hold several questions, and each is read separately. On that sample the first entry's mean is above the board's 1,000 ms limit, because it thinks on more requests than DE-2 did (111 of 700, against DE-2's 80); the second entry is well inside it. If the first entry does not pass your latency measurement, please take the second.

## Run

```sh
git clone https://github.com/shisa-ai/shisa-de              # v0.6.0 (82064a3)
git clone https://github.com/apolinario/decision-index      # 62d2f51
cd shisa-de && python -m venv .venv
.venv/bin/pip install -e . -e ../decision-index

# build the suite into ../decision-index/suite-0.3 as the kit's README describes

CUDA_VISIBLE_DEVICES=0 vllm serve shisa-ai/shisa-de-2.1 --host 127.0.0.1 --port 8026 &    # vLLM 0.30.0, no other options

.venv/bin/python scripts/decision_index_run.py --suite-dir ../decision-index/suite-0.3 \
    --endpoint http://127.0.0.1:8026 --model shisa-ai/shisa-de-2.1 --policy repeat-think --out runs/shisa-de-2.1-adaptive
.venv/bin/python scripts/decision_index_run.py --suite-dir ../decision-index/suite-0.3 \
    --endpoint http://127.0.0.1:8026 --model shisa-ai/shisa-de-2.1 --policy repeat --out runs/shisa-de-2.1-repeat
```

`decision_index_run.py` calls the kit's own `run` and `score` with the engine class `scripts.decision_index_engine:ShisaDE2Engine`. To call the kit directly:

```sh
python -m decision_index run --engine scripts.decision_index_engine:ShisaDE2Engine \
    --model shisa-ai/shisa-de-2.1 --option tokenizer=shisa-ai/shisa-de-2.1 \
    --option base_url=http://127.0.0.1:8026 --option max_tokens=262144 \
    --option serving_manifest=manifest.json --option policy=repeat-think \
    --compact --out runs/shisa-de-2.1-adaptive
```

`manifest.json` is any JSON file describing the server; the engine copies it into `environment.json`.

- **Engine:** [`scripts/decision_index_engine.py`](https://github.com/shisa-ai/shisa-de/blob/82064a3/scripts/decision_index_engine.py) is a kit `Engine` subclass over the SDK's `DecisionModel.decide`, unchanged from the DE-2 submission. There is no `/v1/systemone` server; the engine talks to vLLM's `/v1/completions` directly.
  - It renders the model's chat template with thinking off and requests the logprobs of the option codes by token id, then takes a softmax over the listed options. Temperature 1, calibration off.
  - Each question in a request is read separately, one after another.
  - For the thinking read, the model generates up to 1,024 tokens at temperature 0, the client closes the thought, and the option codes are read after it.
- **Fastest path:** `--option policy=repeat` is the fastest path that keeps the repeated question. The SDK contract is in [docs/READOUT-DE2.md](https://github.com/shisa-ai/shisa-de/blob/82064a3/docs/READOUT-DE2.md).
- **Attention kernel:** on SM90 GPUs (H100, H200, H20), vLLM 0.30.0 runs this base model through a FlashAttention 4 path that lowers accuracy. On the RTX PRO 6000 it selects Triton attention and no option is needed. The model card has the details.
- **How the runs were made:** fresh 0.3 runs with the kit at `62d2f51`, no rows carried over, on the model repo's files served from a local directory under the name `shisa-ai/shisa-de-2`. The 140,620 rows were split into 16 interleaved parts, one client per part, with the parts spread over the servers (two RTX PRO 6000 Blackwell for the first entry, one for the second), then concatenated and scored with `score --edition 0.3`. This differs from the sequential shards of the DE-2 submission; on DE-2 the two setups scored 58.89 and 59.01 under `repeat-think`. The SDK was at `23d3bb8` (first entry) and `d85194f` (second); the engine, readout and policy files are byte-identical to v0.6.0, and each `environment.json` records the hashes of the files it ran.
- **Declared limits:** 256 options per question and a 262,144-token context. A request over either is reported as unsupported, with nothing truncated. No request in the suite reached either.
- **Training data:** described on the [model card](https://huggingface.co/shisa-ai/shisa-de-2.1#training). As for DE-2, the training mixture includes examples derived from the upstream BANKING77 training split (`kev-banking-largek`). The families added in DE-2.1 (tool availability, none-of-the-above options, evidence mentions, more hidden-text examples) are authored and contain no benchmark rows, but their design followed our error analysis on the public suite, and the recipe and the eleven seeds were chosen using public-suite results (full runs and a 10K-row subset).
