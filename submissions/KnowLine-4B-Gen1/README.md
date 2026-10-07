# KnowLine-4B-Gen1

[PelaAI/KnowLine-4B-Gen1](https://huggingface.co/PelaAI/KnowLine-4B-Gen1) is Qwen3.5-4B with a LoRA (rank 32) fine-tune merged into bf16 weights (Apache-2.0).

- It reads the state and one question with all its options, and returns the probability of each option's label token at the answer position: one prefill per question, nothing generated.
- The vision tower is the base model's, unchanged; this suite does not use it.

| | |
|---|---|
| Decision Index 0.3 public index | 60.47 (raw 70.42) |
| Requests | 140,178 of 140,178 scoreable, all answered, none unsupported, no errors |
| Request time in `scores.json` | median 124.9 ms, mean 387.7 ms. Measured with 16 client shards in parallel against one or two servers on NVIDIA H20, so it includes queueing; not measured on an RTX PRO 6000 |

## Run

```sh
pip install "sglang==0.5.21" "transformers==5.12.1" requests
git clone https://huggingface.co/PelaAI/KnowLine-4B-Gen1 && cd KnowLine-4B-Gen1      # @ 6fad739
bash serve_knowline.sh PelaAI/KnowLine-4B-Gen1 0 8080     # SGLang (FP8 at load) on :9080, /v1/systemone on :8080
# equivalent manual launch:
#   CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server --model-path PelaAI/KnowLine-4B-Gen1 --served-model-name m --tp 1 \
#     --quantization fp8 --mem-fraction-static 0.72 --mamba-radix-cache-strategy extra_buffer --enable-fp32-lm-head --port 9080 &
#   python knowline_server.py --model PelaAI/KnowLine-4B-Gen1 --backend sglang --url http://127.0.0.1:9080 \
#     --served-model-name m --temperature 1 --workers 16 --port 8080

git clone https://github.com/apolinario/decision-index && cd decision-index    # 62d2f51
python -m decision_index pipeline --engine http --option base_url=http://127.0.0.1:8080 \
    --option model=m --out runs/KnowLine-4B-Gen1
python -m decision_index score --edition 0.3 --results results.jsonl.gz --out rescore   # the published file
```

- **Front end:** `knowline_server.py` (in the model repo) is the `/v1/systemone` front end that produced this run, packaged as one file.
  - It renders the model's chat template with thinking off: the state, then one user turn with the instruction, the question and the options labelled A, B, ...
  - It reads the label-token logprobs right after `Answer:` and returns a softmax over the option labels, temperature 1, with no calibration file.
  - Multi-question requests are scored one prefill per question, from 16 threads, after one warm-up of the shared prefix.
  - Checked against the front end of this run: identical prompts and label tokens on 7,500 rows, and the same choice on all 953 questions of 200 rows through a live engine.
- **Without SGLang:** `python knowline_server.py --model PelaAI/KnowLine-4B-Gen1 --backend hf` serves through transformers in bf16 (slower, no FP8).
- **Settings:** `--mem-fraction-static` only sizes SGLang's KV cache; scores do not depend on it.
- **How the run was made:** the 0.2.1 requests ran with the kit at `87d4650`. The run directory was then resumed under 0.3 with the kit at `62d2f51`, which ran only the 2,638 rebuilt GSM8K requests, as the 0.3 README describes. Re-scoring the published `results.jsonl.gz` with `62d2f51` (`score --edition 0.3`) gives 60.47.
- **Training data:** it includes the train splits of Decision Index benchmarks in their request formats, about a quarter of the training rows, plus other public datasets and synthetic decision tasks. No test row is included, and the training texts were decontaminated against the public suite rows.
