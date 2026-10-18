# KnowLine-26B-A4B-Gen1

[PelaAI/KnowLine-26B-A4B-Gen1](https://huggingface.co/PelaAI/KnowLine-26B-A4B-Gen1) is google/gemma-4-26B-A4B-it with two LoRA (rank 32) fine-tunes applied in sequence, each merged into bf16 weights (Apache-2.0): a first run on the instruction-tuned model, and a short second run on top of it. A readout temperature of 1.25 is then folded into the weights exactly: the final norm weight is divided by 1.25 and `final_logit_softcapping` goes from 30 to 24, so every final logit is divided by 1.25. This changes the probabilities but not the option ranking.

- It reads the state and one question with all its options, and returns the probability of each option's label token at the answer position: one prefill per question, nothing generated.
- Mixture of experts: about 26B parameters in total (25.8B counting the vision tower), about 4B active per token. The tokenizer and chat template are Gemma 4's. The front end is the same as KnowLine-35B-A3B-Gen1. The vision tower is the base model's, unchanged; the text suite does not use it.

| | |
|---|---|
| Decision Index 0.3 public index | 65.76 (raw 73.97) |
| Requests | 140,178 of 140,178 scoreable, all answered, none unsupported, no errors |
| Request time in `scores.json` | median 227.5 ms, mean 823.8 ms. Measured with 10 or 11 client shards in parallel against each of six identical servers (one per NVIDIA H20), so it includes queueing; not measured on an RTX PRO 6000 |
| Request time, one request at a time | median 44.8 ms, mean 84.6 ms, p80 102 ms, over 300 rows of the public suite on one NVIDIA H20 (with the same weights before the final readout temperature was folded in; the temperature does not change the computation); not measured on an RTX PRO 6000 |

## Run

```sh
pip install "sglang==0.5.21" "transformers==5.12.1" requests
git clone https://huggingface.co/PelaAI/KnowLine-26B-A4B-Gen1 && cd KnowLine-26B-A4B-Gen1      # @ 043e8e4
bash serve_knowline.sh PelaAI/KnowLine-26B-A4B-Gen1 0 8080     # SGLang (bf16) on :9080, /v1/systemone on :8080
# equivalent manual launch:
#   CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server --model-path PelaAI/KnowLine-26B-A4B-Gen1 --served-model-name m --tp 1 \
#     --mem-fraction-static 0.85 --enable-fp32-lm-head --port 9080 &
#   python knowline_server.py --model PelaAI/KnowLine-26B-A4B-Gen1 --backend sglang --url http://127.0.0.1:9080 \
#     --served-model-name m --temperature 1 --workers 16 --port 8080

git clone https://github.com/apolinario/decision-index && cd decision-index    # 62d2f51
python -m decision_index pipeline --engine http --option base_url=http://127.0.0.1:8080 \
    --option model=m --out runs/KnowLine-26B-A4B-Gen1
python -m decision_index score --edition 0.3 --results results.jsonl.gz --out rescore   # the published file
```

- **Front end:** `knowline_server.py` (in the model repo) is the `/v1/systemone` front end of this run. It is the same file as in KnowLine-35B-A3B-Gen1; only the model name in its docstring differs.
  - It renders the model's chat template with thinking off: the state, then one user turn with the instruction, the question and the options labelled A, B, ...
  - It reads the label-token logprobs right after `Answer:` and returns a softmax over the option labels, temperature 1, with no calibration file; the readout temperature is in the weights.
  - Multi-question requests are scored one prefill per question, from 16 threads, after one warm-up of the shared prefix.
- **Settings:** the weights are served in bf16. FP8 at load was not stable from run to run for this model (95.7% of decisions agreed between two identical FP8 runs, against 99.35% in bf16), so the run and `serve_knowline.sh` use bf16. `--mem-fraction-static` only sizes SGLang's KV cache; scores do not depend on it. One 96 GB GPU is enough (the bf16 weights are about 52 GB).
- **Weights:** the run (2026-10-18) served the local copy of the released weights before the upload. All files in the model repo @ `043e8e4` other than the model cards match `SHA256SUMS` there.
- **How the run was made:** a fresh 0.3 run with the kit at `62d2f51`, no rows carried over. The 140,620 rows that `decision-index run --edition 0.3` sends were split round-robin into 64 shards, each run as `decision-index run --edition 0.3 --rows <shard>` in parallel against six identical servers (one per GPU), then concatenated and scored with `score --edition 0.3`, which gives 65.76.
- **Training data:** it includes the train splits of Decision Index benchmarks in their request formats: in the first run about 36% of the training rows target Decision Index benchmarks in their request format or a reworded version of it (train splits of public datasets and synthetic items written in the same format), plus 0.7% copies of the same items rendered in other formats. The second run is one third injection-invariance items (instructions planted in the state that must not change the answer) and two thirds a replay of the first run. The rest is other public datasets and synthetic decision tasks. For BPoMP and cfcolor, training items come from the same source datasets as the benchmarks, with every benchmark item excluded, as in KnowLine-4B and KnowLine-35B-A3B-Gen1.
- **Decontamination:** all training data was checked against every public suite row with the same check as our earlier releases: near-duplicates, repeated questions or claims and whole states were removed (491 rows in the parts built for this release; the replayed parts come from data that had already passed the same check); rows that only share a source passage (for example a Wikipedia paragraph or a tool description) were kept.
