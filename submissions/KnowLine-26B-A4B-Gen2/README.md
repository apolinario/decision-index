# KnowLine-26B-A4B-Gen2

[PelaAI/KnowLine-26B-A4B-Gen2](https://huggingface.co/PelaAI/KnowLine-26B-A4B-Gen2) is google/gemma-4-26B-A4B-it with three LoRA (rank 32) fine-tunes applied in sequence, each merged into bf16 weights (Apache-2.0): a first run on the instruction-tuned model, a short second run on top of it, and a third run on more rows from the same sources. It is the next version of KnowLine-26B-A4B-Gen1. A readout temperature of 1.25 is then folded into the weights exactly: the final norm weight is divided by 1.25 and `final_logit_softcapping` goes from 30 to 24, so every final logit is divided by 1.25. This changes the probabilities but not the option ranking.

- It reads the state and one question with all its options, and returns the probability of each option's label token at the answer position: one prefill per question, nothing generated.
- Mixture of experts: about 26B parameters in total (25.8B counting the vision tower), about 4B active per token. The tokenizer and chat template are Gemma 4's. The front end is the same as KnowLine-35B-A3B-Gen1. The vision tower is the base model's, unchanged; the text suite does not use it.

| | |
|---|---|
| Decision Index 0.3 public index | 66.94 (raw 74.94) |
| Requests | 140,178 of 140,178 scoreable, all answered, none unsupported, no errors |
| Request time in `scores.json` | median 228.7 ms, mean 820.7 ms. Measured with 10 or 11 client shards in parallel against each of six identical servers (one per NVIDIA H20), so it includes queueing; not measured on an RTX PRO 6000 |
| Request time, one request at a time | median 43.1 ms, mean 82.8 ms, p80 99 ms, over 300 rows of the public suite on one NVIDIA H20; not measured on an RTX PRO 6000 |

## Run

```sh
pip install "sglang==0.5.21" "transformers==5.12.1" requests
git clone https://huggingface.co/PelaAI/KnowLine-26B-A4B-Gen2 && cd KnowLine-26B-A4B-Gen2      # @ 284787c
bash serve_knowline.sh PelaAI/KnowLine-26B-A4B-Gen2 0 8080     # SGLang (bf16) on :9080, /v1/systemone on :8080
# equivalent manual launch:
#   CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server --model-path PelaAI/KnowLine-26B-A4B-Gen2 --served-model-name m --tp 1 \
#     --mem-fraction-static 0.85 --enable-fp32-lm-head --port 9080 &
#   python knowline_server.py --model PelaAI/KnowLine-26B-A4B-Gen2 --backend sglang --url http://127.0.0.1:9080 \
#     --served-model-name m --temperature 1 --workers 16 --port 8080

git clone https://github.com/apolinario/decision-index && cd decision-index    # 62d2f51
python -m decision_index pipeline --engine http --option base_url=http://127.0.0.1:8080 \
    --option model=m --out runs/KnowLine-26B-A4B-Gen2
python -m decision_index score --edition 0.3 --results results.jsonl.gz --out rescore   # the published file
```

- **Front end:** `knowline_server.py` (in the model repo) is the `/v1/systemone` front end of this run. It is the same file as in KnowLine-35B-A3B-Gen1; only the model name in its docstring differs.
  - It renders the model's chat template with thinking off: the state, then one user turn with the instruction, the question and the options labelled A, B, ...
  - It reads the label-token logprobs right after `Answer:` and returns a softmax over the option labels, temperature 1, with no calibration file; the readout temperature is in the weights.
  - Multi-question requests are scored one prefill per question, from 16 threads, after one warm-up of the shared prefix.
- **Settings:** the weights are served in bf16. FP8 at load was not stable from run to run for this model (95.7% of decisions agreed between two identical FP8 runs, against 99.35% in bf16), so the run and `serve_knowline.sh` use bf16. `--mem-fraction-static` only sizes SGLang's KV cache; scores do not depend on it. One 96 GB GPU is enough (the bf16 weights are about 52 GB).
- **Weights:** the run (2026-10-19) served the local copy of the released weights before the upload. All files in the model repo @ `284787c` other than the model cards match `SHA256SUMS` there.
- **How the run was made:** a fresh 0.3 run with the kit at `62d2f51`, no rows carried over. The 140,620 rows that `decision-index run --edition 0.3` sends were split round-robin into 64 shards, each run as `decision-index run --edition 0.3 --rows <shard>` in parallel against six identical servers (one per GPU), then concatenated and scored with `score --edition 0.3`, which gives 66.94.
- **Training data:** it includes the train splits of Decision Index benchmarks in their request formats. In the first run about 34% of the training rows target Decision Index benchmarks in their request format or a reworded version of it (train splits of public datasets and synthetic items written in the same format), plus 0.5% copies of the same items rendered in other formats. The second run is one third injection-invariance items (instructions planted in the state that must not change the answer) and two thirds a replay of the first run. The third run is 250k more rows from the same sources as the earlier KnowLine training data plus replays of the first two runs; about 26% of its rows target Decision Index benchmarks in the same sense, plus 3.1% format copies. The rest is other public datasets and synthetic decision tasks. For BPoMP and cfcolor, training items come from the same source datasets as the benchmarks, with every benchmark item excluded, as in KnowLine-4B and KnowLine-35B-A3B-Gen1.
- **Decontamination:** all training data was checked against every public suite row with the same check as our earlier releases: near-duplicates, repeated questions or claims and whole states were removed (491 rows in the parts built for Gen1 and 751 in the third run; replayed parts come from data that had already passed the same check); rows that only share a source passage (for example a Wikipedia paragraph or a tool description) were kept.
