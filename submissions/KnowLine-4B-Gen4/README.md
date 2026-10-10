# KnowLine-4B-Gen4

[PelaAI/KnowLine-4B-Gen4](https://huggingface.co/PelaAI/KnowLine-4B-Gen4) is a weight average of LoRA (rank 32) fine-tunes of Qwen3.5-4B, each merged into bf16 weights (Apache-2.0): 0.5 × [KnowLine-4B-Gen3](https://huggingface.co/PelaAI/KnowLine-4B-Gen3) + 0.5 × Gen3 trained further in a fourth run. The final norm weight then folds in a readout temperature of 1.32 (every logit divided by 1.32), which changes the probabilities but not the option ranking.

- It reads the state and one question with all its options, and returns the probability of each option's label token at the answer position: one prefill per question, nothing generated.
- Architecture, config, tokenizer, chat template and front end are the same as KnowLine-4B-Gen1-Gen3. The vision tower is the base model's, unchanged; this suite does not use it.

| | |
|---|---|
| Decision Index 0.3 public index | 64.90 (raw 73.58) |
| Requests | 140,178 of 140,178 scoreable, all answered, none unsupported, no errors |
| Request time in `scores.json` | median 200.6 ms, mean 606.9 ms. Measured with 16 client shards in parallel against each of four identical servers (one per NVIDIA H20), so it includes queueing; not measured on an RTX PRO 6000 |

## Run

```sh
pip install "sglang==0.5.21" "transformers==5.12.1" requests
git clone https://huggingface.co/PelaAI/KnowLine-4B-Gen4 && cd KnowLine-4B-Gen4      # @ 5482f96
bash serve_knowline.sh PelaAI/KnowLine-4B-Gen4 0 8080     # SGLang (FP8 at load) on :9080, /v1/systemone on :8080
# equivalent manual launch:
#   CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server --model-path PelaAI/KnowLine-4B-Gen4 --served-model-name m --tp 1 \
#     --quantization fp8 --mem-fraction-static 0.72 --mamba-radix-cache-strategy extra_buffer --enable-fp32-lm-head --port 9080 &
#   python knowline_server.py --model PelaAI/KnowLine-4B-Gen4 --backend sglang --url http://127.0.0.1:9080 \
#     --served-model-name m --temperature 1 --workers 16 --port 8080

git clone https://github.com/apolinario/decision-index && cd decision-index    # 62d2f51
python -m decision_index pipeline --engine http --option base_url=http://127.0.0.1:8080 \
    --option model=m --out runs/KnowLine-4B-Gen4
python -m decision_index score --edition 0.3 --results results.jsonl.gz --out rescore   # the published file
```

- **Front end:** `knowline_server.py` (in the model repo) is the `/v1/systemone` front end of this run, packaged as one file. It is the same file as in KnowLine-4B-Gen1-Gen3; only the model name in its docstring differs.
  - It renders the model's chat template with thinking off: the state, then one user turn with the instruction, the question and the options labelled A, B, ...
  - It reads the label-token logprobs right after `Answer:` and returns a softmax over the option labels, temperature 1, with no calibration file; the readout temperature is in the weights.
  - Multi-question requests are scored one prefill per question, from 16 threads, after one warm-up of the shared prefix.
  - Checked against our in-house front end, which served this run: identical prompts and label tokens on 7,500 rows, and the same choice on all 953 questions of 200 rows through a live engine (checked on Gen1).
- **Settings:** `--mem-fraction-static` only sizes SGLang's KV cache; scores do not depend on it.
- **How the run was made:** a fresh 0.3 run with the kit at `62d2f51`, no rows carried over. The 140,620 rows that `decision-index run --edition 0.3` sends were split round-robin into 64 shards, each run as `decision-index run --edition 0.3 --rows <shard>` in parallel, 16 against each of four identical servers (one per GPU), then concatenated and scored with `score --edition 0.3`. Re-scoring the published `results.jsonl.gz` with `62d2f51` gives 64.90.
- **Training data:** it includes the train splits of Decision Index benchmarks in their request formats: about a quarter of the training rows in the first run, about 31% in the second, about a quarter in the third and about 34% in the fourth, plus other public datasets and synthetic decision tasks. For BPoMP and cfcolor, training items come from the same source datasets as the benchmarks, with every benchmark item excluded; an item-level check found no shared item.
- **Decontamination:** the fourth run's data was checked against every public suite row with a revised check: near-duplicates, repeated questions or claims and whole states were removed; rows that only share a source passage (for example a Wikipedia paragraph or a tool description) were kept.
