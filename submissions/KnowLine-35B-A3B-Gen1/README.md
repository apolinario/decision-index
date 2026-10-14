# KnowLine-35B-A3B-Gen1

[PelaAI/KnowLine-35B-A3B-Gen1](https://huggingface.co/PelaAI/KnowLine-35B-A3B-Gen1) is a weight average (0.5 / 0.5) of two LoRA (rank 32) fine-tunes of Qwen3.5-35B-A3B, each merged into bf16 weights (Apache-2.0): a first run from the base model, and that run trained further in a second run. The final norm weight then folds in a readout temperature of 1.20 (every logit divided by 1.20), which changes the probabilities but not the option ranking.

- It reads the state and one question with all its options, and returns the probability of each option's label token at the answer position: one prefill per question, nothing generated.
- Mixture of experts: 35B parameters in total, about 3B active per token. Tokenizer, chat template and front end are the same as KnowLine-4B. The vision tower and MTP head are the base model's, unchanged; the text suite does not use them, the vision suite ([Vision](#vision)) uses the vision tower.

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

## Vision

| | |
|---|---|
| Decision Index 0.3 public vision index | 73.00 |
| Rows | 12,210 of 12,210, all answered, none unsupported, no errors; `vision-scores.json` says `"complete": true` |
| Request time, one request at a time | median 261 ms, mean 292 ms, p80 389 ms (max 1,170 ms), over 30 evenly spaced rows of each benchmark (330 rows). Measured on two NVIDIA H20 with tensor parallel 2 while a training job was running on the same GPUs; not measured on an RTX PRO 6000 |
| Request time in the full run | median 1,306 ms, mean 1,596 ms: 16 client workers against the one server, so it includes queueing |

| Benchmark | Rows | Accuracy | Skill |
|---|---:|---:|---:|
| CV-Bench | 2,038 | 85.62 | 76.28 |
| BLINK | 961 | 73.15 | 58.44 |
| RealWorldQA | 589 | 82.68 | 71.70 |
| CharXiv | 532 | 84.40 | 79.20 |
| InfographicVQA | 244 | 99.18 | 98.91 |
| Mind2Web | 973 | 84.89 | 79.86 |
| Winoground | 800 | 90.12 | 80.25 |
| KIE (CORD+FUNSD) | 2,465 | 95.74 | 92.99 |
| Moderation (Hateful Memes) | 1,213 | 73.87 | 47.73 |
| R-Bench-M | 665 | 41.95 | 30.34 |
| MMMU-Pro vision | 1,730 | 52.08 | 45.04 |

```sh
git clone https://github.com/apolinario/decision-index && cd decision-index    # f5a6e4f
pip install -e '.[vision]'
hf auth login                                       # Winoground is gated: accept its terms on the Hub first
decision-index suite rebuild --vision --work work --font "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
decision-index vision verify --dir work/vision      # all 12 sets and 11 benchmarks match

# in the model repo @ e800968; one free 96 GB GPU is enough: bash serve_knowline.sh PelaAI/KnowLine-35B-A3B-Gen1 0 8080
# this run, on two GPUs shared with a training job:
CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server --model-path PelaAI/KnowLine-35B-A3B-Gen1 --served-model-name m --tp 2 \
    --quantization fp8 --mem-fraction-static 0.5 --mamba-radix-cache-strategy extra_buffer --enable-fp32-lm-head --port 9080 &
python knowline_server.py --model PelaAI/KnowLine-35B-A3B-Gen1 --backend sglang --url http://127.0.0.1:9080 \
    --served-model-name m --temperature 1 --workers 16 --port 8080

PYTHONPATH=submissions/KnowLine-35B-A3B-Gen1 decision-index vision pipeline --dir work/vision \
    --engine knowline_vision_engine:KnowLineVision --option base_url=http://127.0.0.1:8080 --option model=m \
    --workers 16 --out runs/KnowLine-35B-A3B-Gen1-vision --fresh
decision-index vision score --dir work/vision --results results.jsonl --out rescore   # the published file, gunzipped
```

- **Results:** [`runs/KnowLine-35B-A3B-Gen1-vision`](https://huggingface.co/datasets/PelaAI/KnowLine-35B-A3B-Gen1-decision-index/tree/467761bf889d653bc0673a2c96584694e889de71/runs/KnowLine-35B-A3B-Gen1-vision) in the same results dataset @ `467761b`: `results.jsonl.gz`, `vision-scores.json`, `environment.json`, `status.json`, and the engine module. Re-scoring the published `results.jsonl.gz` with the kit at `f5a6e4f` gives 73.00. The latency pass is in [`runs/KnowLine-35B-A3B-Gen1-vision-latency`](https://huggingface.co/datasets/PelaAI/KnowLine-35B-A3B-Gen1-decision-index/tree/467761bf889d653bc0673a2c96584694e889de71/runs/KnowLine-35B-A3B-Gen1-vision-latency).
- **Engine:** `knowline_vision_engine.py` (this folder) subclasses the kit's `HttpDecide`, keeping its retries, image cap and latency timing, and changes only the request body. `knowline_server.py` takes images as OpenAI-style content parts inside a chat state, so the row's images go into one user message of the state, in the row's order. The question and its lettered options stay a normal `choice` question, and the answer is read from `answers.answer.probabilities`, as in the kit's `decisions` protocol.
- **Readout:** the same as for text, one prefill per question, reading the option labels' probabilities at the model's usual `Answer:` position. The suite's `prompt.readout` hint (`"The answer is ("` for R-Bench-M and MMMU-Pro vision) is not used; for MMMU-Pro vision the options are the letters themselves (`options_in_image`) and are sent as they are.
- **Image resolution:** native resolution capped at 1.6 MP. The kit downscales images over 1,638,400 pixels (aspect kept) and sends smaller ones as their original bytes. The model's processor (`preprocessor_config.json`, `size.longest_edge` 16,777,216) allows more than 1.6 MP, so there is no lower cap to lift. `environment.json` records `max_image_pixels` 1,638,400.
- **Server settings:** the run used tensor parallel 2 and `--mem-fraction-static 0.5` only because a training job held about 36 GB on every GPU. `--mem-fraction-static` only sizes SGLang's KV cache, and the image scores do not depend on it.
- **How the run was made:** a fresh run (`--fresh`) with the kit at `f5a6e4f` and verification on. The suite was rebuilt with the pinned sources and `vision verify` passed for all 12 sets and 11 benchmarks: Mind2Web labels drawn in Arial Bold, Winoground from the gated `facebook/winoground` after accepting its terms. The latency pass used the same server, one worker, on 30 evenly spaced rows of each benchmark taken from `vision-rows.jsonl`.
- **Training data:** no image training; the model was trained on text only. Its only data from a vision suite source is text-only rows built from the Mind2Web **train** split: cleaned-HTML candidate elements and operations, no screenshots: 6,020 rows (0.4%) in the first run and 1,231 (0.1%) in the second. The suite's Mind2Web rows come from the `test_website` split, whose websites are not in the train split.
