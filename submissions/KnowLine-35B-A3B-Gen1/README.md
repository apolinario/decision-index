# KnowLine-35B-A3B-Gen1

[PelaAI/KnowLine-35B-A3B-Gen1](https://huggingface.co/PelaAI/KnowLine-35B-A3B-Gen1) is a weight average (0.5 / 0.5) of two LoRA (rank 32) fine-tunes of Qwen3.5-35B-A3B, each merged into bf16 weights (Apache-2.0): a first run from the base model, and that run trained further in a second run. The final norm weight then folds in a readout temperature of 1.20 (every logit divided by 1.20), which changes the probabilities but not the option ranking.

- It reads the state and one question with all its options, and returns the probability of each option's label token at the answer position: one prefill per question, nothing generated.
- Mixture of experts: 35B parameters in total, about 3B active per token. Tokenizer, chat template and front end are the same as KnowLine-4B. The vision tower and MTP head are the base model's, unchanged; this suite does not use them.

| | |
|---|---|
| Decision Index 0.3 public index | 69.49 (raw 76.86) |
| Requests | 140,178 of 140,178 scoreable, all answered, none unsupported, no errors |
| Request time in `scores.json` | median 229.7 ms, mean 650.8 ms. Measured with 16 client shards in parallel against each of four identical servers (one per NVIDIA H20), so it includes queueing; not measured on an RTX PRO 6000 |

## Run

```sh
pip install "sglang==0.5.21" "transformers==5.12.1" requests
git clone https://huggingface.co/PelaAI/KnowLine-35B-A3B-Gen1 && cd KnowLine-35B-A3B-Gen1      # @ e800968
bash serve_knowline.sh PelaAI/KnowLine-35B-A3B-Gen1 0 8080     # SGLang (FP8 at load, about 35 GB) on :9080, /v1/systemone on :8080
# equivalent manual launch:
#   CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server --model-path PelaAI/KnowLine-35B-A3B-Gen1 --served-model-name m --tp 1 \
#     --quantization fp8 --mem-fraction-static 0.72 --mamba-radix-cache-strategy extra_buffer --enable-fp32-lm-head --port 9080 &
#   python knowline_server.py --model PelaAI/KnowLine-35B-A3B-Gen1 --backend sglang --url http://127.0.0.1:9080 \
#     --served-model-name m --temperature 1 --workers 16 --port 8080

git clone https://github.com/apolinario/decision-index && cd decision-index    # 62d2f51
python -m decision_index pipeline --engine http --option base_url=http://127.0.0.1:8080 \
    --option model=m --out runs/KnowLine-35B-A3B-Gen1
python -m decision_index score --edition 0.3 --results results.jsonl.gz --out rescore   # the published file
```

- **Front end:** `knowline_server.py` (in the model repo) is the `/v1/systemone` front end of this run, packaged as one file. It is the same file as in KnowLine-4B-Gen4; only the model name in its docstring differs.
  - It renders the model's chat template with thinking off: the state, then one user turn with the instruction, the question and the options labelled A, B, ...
  - It reads the label-token logprobs right after `Answer:` and returns a softmax over the option labels, temperature 1, with no calibration file; the readout temperature is in the weights.
  - Multi-question requests are scored one prefill per question, from 16 threads, after one warm-up of the shared prefix.
  - The run itself was served by our in-house front end, which `knowline_server.py` packages; the check between the two is described in the KnowLine-4B-Gen4 submission.
- **Settings:** `--mem-fraction-static` only sizes SGLang's KV cache; scores do not depend on it. One 96 GB GPU is enough; plan for at least 48 GB.
- **Weights:** the run (2026-10-11) served the local copy of the released weights before the upload. All weight, config and tokenizer files match `SHA256SUMS` in the model repo; the commits after the first weights commit `d717020` change only the model cards.
- **How the run was made:** a fresh 0.3 run with the kit at `62d2f51`, no rows carried over. The 140,620 rows that `decision-index run --edition 0.3` sends were split round-robin into 64 shards, each run as `decision-index run --edition 0.3 --rows <shard>` in parallel, 16 against each of four identical servers (one per GPU), then concatenated and scored with `score --edition 0.3`. Re-scoring the published `results.jsonl.gz` with `62d2f51` gives 69.49.
- **Training data:** it includes the train splits of Decision Index benchmarks in their request formats: in the first run about 21% of the training rows, plus 4% copies of the same items rendered in other formats; in the second run about 8% and 9%. The rest is other public datasets and synthetic decision tasks. For BPoMP and cfcolor, training items come from the same source datasets as the benchmarks, with every benchmark item excluded, as in KnowLine-4B.
- **Decontamination:** all training data of both runs was checked against every public suite row with the same check as KnowLine-4B-Gen4: near-duplicates, repeated questions or claims and whole states were removed (7,833 of 1,500,018 rows in the first run, 2,445 of 985,519 in the second); rows that only share a source passage (for example a Wikipedia paragraph or a tool description) were kept.
