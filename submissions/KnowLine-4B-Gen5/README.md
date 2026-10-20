# KnowLine-4B-Gen5

[PelaAI/KnowLine-4B-Gen5](https://huggingface.co/PelaAI/KnowLine-4B-Gen5) is a weight average (0.5 / 0.5) of two LoRA (rank 32) fine-tunes of Qwen3.5-4B, each merged into bf16 weights (Apache-2.0): the KnowLine-4B line trained further on newer data, and a model trained from the base model on a new data round. The final norm weight then folds in a readout temperature of 1.50 (every logit divided by 1.50), which changes the probabilities but not the option ranking. It is the next version of [KnowLine-4B-Gen4](https://huggingface.co/PelaAI/KnowLine-4B-Gen4).

- It reads the state and one question with all its options, and returns the probability of each option's label token at the answer position: one prefill per question, nothing generated.
- Architecture, config, tokenizer, chat template and front end are the same as KnowLine-4B-Gen1-Gen4. The vision tower is the base model's, unchanged; this suite does not use it.

| | |
|---|---|
| Decision Index 0.3 public index | 65.67 (raw 74.14) |
| Requests | 140,178 of 140,178 scoreable, all answered, none unsupported, no errors |
| Request time in `scores.json` | median 205.7 ms, mean 608.6 ms. Measured with 16 client shards in parallel against each of four identical servers (one per NVIDIA H20), so it includes queueing; not measured on an RTX PRO 6000 |
| Request time, one request at a time | median 31.6 ms, mean 58.8 ms, p80 57 ms, over 300 rows of the public suite on one NVIDIA H20; not measured on an RTX PRO 6000 |

## Run

```sh
pip install "sglang==0.5.21" "transformers==5.12.1" requests
git clone https://huggingface.co/PelaAI/KnowLine-4B-Gen5 && cd KnowLine-4B-Gen5      # @ a8e0cd6
bash serve_knowline.sh PelaAI/KnowLine-4B-Gen5 0 8080     # SGLang (FP8 at load) on :9080, /v1/systemone on :8080
# equivalent manual launch:
#   CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server --model-path PelaAI/KnowLine-4B-Gen5 --served-model-name m --tp 1 \
#     --quantization fp8 --mem-fraction-static 0.72 --mamba-radix-cache-strategy extra_buffer --enable-fp32-lm-head --port 9080 &
#   python knowline_server.py --model PelaAI/KnowLine-4B-Gen5 --backend sglang --url http://127.0.0.1:9080 \
#     --served-model-name m --temperature 1 --workers 16 --port 8080

git clone https://github.com/apolinario/decision-index && cd decision-index    # 62d2f51
python -m decision_index pipeline --engine http --option base_url=http://127.0.0.1:8080 \
    --option model=m --out runs/KnowLine-4B-Gen5
python -m decision_index score --edition 0.3 --results results.jsonl.gz --out rescore   # the published file
```

- **Front end:** `knowline_server.py` (in the model repo) is the `/v1/systemone` front end of this run. It is the same file as in KnowLine-4B-Gen4; only the model name in its docstring differs.
  - It renders the model's chat template with thinking off: the state, then one user turn with the instruction, the question and the options labelled A, B, ...
  - It reads the label-token logprobs right after `Answer:` and returns a softmax over the option labels, temperature 1, with no calibration file; the readout temperature is in the weights.
  - Multi-question requests are scored one prefill per question, from 16 threads, after one warm-up of the shared prefix.
- **Settings:** `--mem-fraction-static` only sizes SGLang's KV cache; scores do not depend on it.
- **Weights:** the run (2026-10-21) served the local copy of the released weights before the upload. All files in the model repo @ `a8e0cd6` other than the model cards match `SHA256SUMS` there.
- **How the run was made:** a fresh 0.3 run with the kit at `62d2f51`, no rows carried over. The 140,620 rows that `decision-index run --edition 0.3` sends were split round-robin into 64 shards, each run as `decision-index run --edition 0.3 --rows <shard>` in parallel, 16 against each of four identical servers (one per GPU), then concatenated and scored with `score --edition 0.3`, which gives 65.67.
- **Training data:** it includes the train splits of Decision Index benchmarks in their request formats. In the data rounds added for this model, about 24% of the training rows target Decision Index benchmarks in their request format or a reworded version of it (train splits of public datasets and synthetic items written in the same format), plus 1.4% copies of the same items rendered in other formats; the rest is other public datasets and synthetic decision tasks. The earlier KnowLine-4B rounds are described in the KnowLine-4B-Gen4 submission. For BPoMP and cfcolor, training items come from the same source datasets as the benchmarks, with every benchmark item excluded.
- **Decontamination:** all training data of this round, new and replayed, was checked against every public suite row and our other evaluation sets with the same tool as our earlier releases. About 22,000 of 3.1 million rows were removed: near-duplicates, repeated questions or claims and whole states, and, in a stricter second pass, most rows that share a source passage with an evaluation item.
