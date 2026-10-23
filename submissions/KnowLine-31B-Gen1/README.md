# KnowLine-31B-Gen1

[PelaAI/KnowLine-31B-Gen1](https://huggingface.co/PelaAI/KnowLine-31B-Gen1) is google/gemma-4-31B-it with one LoRA (rank 32) fine-tune merged into bf16 weights, then averaged with the base model tensor by tensor: 0.75 × fine-tuned + 0.25 × gemma-4-31B-it (Apache-2.0). A readout temperature of 1.44 is then folded into the weights exactly: the final norm weight is divided by 1.44 and `final_logit_softcapping` goes from 30 to 20.83, so every final logit is divided by 1.44. This changes the probabilities but not the option ranking.

- It reads the state and one question with all its options, and returns the probability of each option's label token at the answer position: one prefill per question, nothing generated.
- Dense, about 31B parameters (31.3B counting the vision tower). The tokenizer and chat template are Gemma 4's. The vision tower is the base model's, unchanged; the text suite does not use it.
- The front end reads short requests twice: a text-only request of at most 3,000 tokens is written a second time, introduced by "Read the same input again before answering:", before the answer is read; longer requests are sent once. In this run 132,376 of the 140,620 requests were read twice. It is the front end of KnowLine-26B-A4B-Gen2.5 and KnowLine-35B-A3B-Gen1.5; only the model name in its docstrings differs.

| | |
|---|---|
| Decision Index 0.3 public index | 70.13 (raw 77.07) |
| Requests | complete: 140,178 of 140,178 scoreable requests, all answered, none unsupported, nothing truncated, no errors |
| Request time in `scores.json` | median 1317.3 ms, mean 3900.0 ms. Measured with 4 client shards in parallel against each of four identical servers (one per NVIDIA H20), so it includes queueing; not measured on an RTX PRO 6000 |
| Request time, one request at a time | median 340.3 ms, mean 910.0 ms, p80 1185 ms, over 300 rows of the public suite on one NVIDIA H20; not measured on an RTX PRO 6000 |

## Run

```sh
pip install "sglang==0.5.21" "transformers==5.12.1" requests
git clone https://huggingface.co/PelaAI/KnowLine-31B-Gen1 && cd KnowLine-31B-Gen1      # @ 1fc9d4a
bash serve_knowline.sh PelaAI/KnowLine-31B-Gen1 0 8080     # SGLang (bf16) on :9080, /v1/systemone on :8080 (read twice on)
# equivalent manual launch:
#   CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server --model-path PelaAI/KnowLine-31B-Gen1 --served-model-name m --tp 1 \
#     --mem-fraction-static 0.85 --enable-fp32-lm-head --port 9080 &
#   python knowline_server.py --model PelaAI/KnowLine-31B-Gen1 --backend sglang --url http://127.0.0.1:9080 \
#     --served-model-name m --temperature 1 --workers 16 --port 8080 --temperatures temperatures.json

git clone https://github.com/apolinario/decision-index && cd decision-index    # 62d2f51
python -m decision_index pipeline --engine http --option base_url=http://127.0.0.1:8080 \
    --option model=m --out runs/KnowLine-31B-Gen1
python -m decision_index score --edition 0.3 --results results.jsonl.gz --out rescore   # the published file
```

- **Front end:** `knowline_server.py` with `knowline_layout.py` (both in the model repo) is the `/v1/systemone` front end of this run.
  - It renders the model's chat template with thinking off: the state, then one user turn with the instruction, the question and the options labelled A, B, ...; with read twice, the whole input is then written once more before the answer.
  - It reads the label-token logprobs right after `Answer:` and returns a softmax over the option labels, temperature 1, with no calibration file; the readout temperature is in the weights. With `--temperatures temperatures.json` (`{"score": 0.6944}`, in the model repo; `serve_knowline.sh` passes it and this run used it), the logits of score (rubric) questions are divided by a further 0.6944, an effective temperature of 1.0 for them; this changes their probabilities, not which option ranks first.
  - Multi-question requests are scored one prefill per question, from 16 threads, after one warm-up of the shared prefix.
- **Settings:** the weights are served in bf16, as for our Gemma 4 26B-A4B models. `--mem-fraction-static` only sizes SGLang's KV cache; scores do not depend on it. One 96 GB GPU per server (the bf16 weights are about 63 GB).
- **Weights:** the run (2026-10-24) served the local copy of the released weights. All files in the model repo @ `1fc9d4a` other than the model cards match `SHA256SUMS` there.
- **How the run was made:** a fresh 0.3 run with the kit at `62d2f51`, no rows carried over. The 140,620 rows that `decision-index run --edition 0.3` sends were split round-robin into 16 shards, each run as `decision-index run --edition 0.3 --rows <shard>` in parallel against four identical servers (one per GPU), then concatenated and scored with `score --edition 0.3`, which gives 70.13.
- **Training data:** the training data of KnowLine-26B-A4B-Gen1, in one pass (906,122 rows). It includes the train splits of Decision Index benchmarks in their request formats: about 33% of the training rows target Decision Index benchmarks in their request format or a reworded version of it (train splits of public datasets and synthetic items written in the same format), plus 0.5% copies of the same items rendered in other formats, and 2.6% injection-invariance items (instructions planted in the state that must not change the answer). The rest is other public datasets and synthetic decision tasks. For BPoMP and cfcolor, training items come from the same source datasets as the benchmarks, with every benchmark item excluded, as in our earlier models.
- **Decontamination:** all training data was checked against every public suite row with the same check as our earlier releases: near-duplicates, repeated questions or claims and whole states were removed (491 rows when this data was built for KnowLine-26B-A4B-Gen1); rows that only share a source passage (for example a Wikipedia paragraph or a tool description) were kept.

## Vision

| | |
|---|---|
| Decision Index 0.3 public vision index | 69.30 |
| Rows | 12,210 of 12,210, all answered, none unsupported, no errors; `vision-scores.json` says `"complete": true` |
| Request time, one request at a time | median 292 ms, mean 344 ms, p80 353 ms (max 1,230 ms), over 30 evenly spaced rows of each benchmark (330 rows), on one NVIDIA H20; not measured on an RTX PRO 6000 |
| Request time in the full run | median 3,648 ms, mean 3,656 ms: 16 client workers against the one server, so it includes queueing |

| Benchmark | Rows | Accuracy | Skill |
|---|---:|---:|---:|
| CV-Bench | 2,038 | 84.30 | 74.10 |
| BLINK | 961 | 72.84 | 57.95 |
| RealWorldQA | 589 | 73.01 | 55.88 |
| CharXiv | 532 | 81.02 | 74.69 |
| InfographicVQA | 244 | 93.44 | 91.26 |
| Mind2Web | 973 | 80.06 | 73.42 |
| Winoground | 800 | 93.00 | 86.00 |
| KIE (CORD+FUNSD) | 2,465 | 96.19 | 93.72 |
| Moderation (Hateful Memes) | 1,213 | 72.22 | 44.44 |
| R-Bench-M | 665 | 41.95 | 30.34 |
| MMMU-Pro vision | 1,730 | 44.39 | 36.22 |

```sh
git clone https://github.com/apolinario/decision-index && cd decision-index    # f5a6e4f
pip install -e '.[vision]'
hf auth login                                       # Winoground is gated: accept its terms on the Hub first
decision-index suite rebuild --vision --work work --font "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
decision-index vision verify --dir work/vision      # all 12 sets and 11 benchmarks match

# in the model repo @ 1fc9d4a; one 96 GB GPU:
bash serve_knowline.sh PelaAI/KnowLine-31B-Gen1 0 8080

PYTHONPATH=submissions/KnowLine-31B-Gen1 decision-index vision pipeline --dir work/vision \
    --engine knowline_vision_engine:KnowLineVision --option base_url=http://127.0.0.1:8080 --option model=m \
    --workers 16 --out runs/KnowLine-31B-Gen1-vision --fresh
decision-index vision score --dir work/vision --results results.jsonl --out rescore   # the published file, gunzipped
```

- **Results:** [`runs/KnowLine-31B-Gen1-vision`](https://huggingface.co/datasets/PelaAI/KnowLine-31B-Gen1-decision-index/tree/bb63d2442f8336a2184e7df8659b97ab921ab81a/runs/KnowLine-31B-Gen1-vision) in the same results dataset @ `bb63d24`: `results.jsonl.gz`, `vision-scores.json`, `environment.json`, `status.json`, and the engine module. Re-scoring the published `results.jsonl.gz` with the kit at `f5a6e4f` gives 69.30. The latency pass is in [`runs/KnowLine-31B-Gen1-vision-latency`](https://huggingface.co/datasets/PelaAI/KnowLine-31B-Gen1-decision-index/tree/bb63d2442f8336a2184e7df8659b97ab921ab81a/runs/KnowLine-31B-Gen1-vision-latency).
- **Engine:** `knowline_vision_engine.py` (this folder) subclasses the kit's `HttpDecide`, keeping its retries, image cap and latency timing, and changes only the request body. `knowline_server.py` takes images as OpenAI-style content parts inside a chat state, so the row's images go into one user message of the state, in the row's order. The question and its lettered options stay a normal `choice` question, and the answer is read from `answers.answer.probabilities`, as in the kit's `decisions` protocol. It is the engine of KnowLine-35B-A3B-Gen1's vision run.
- **Readout:** the same as for text, one prefill per question, reading the option labels' probabilities at the model's usual `Answer:` position. Requests with images are read once (read twice applies only to text-only requests). The suite's `prompt.readout` hint (`"The answer is ("` for R-Bench-M and MMMU-Pro vision) is not used; for MMMU-Pro vision the options are the letters themselves (`options_in_image`) and are sent as they are.
- **Image resolution:** the kit downscales images over 1,638,400 pixels (aspect kept) and sends smaller ones as their original bytes. Gemma 4's processor then encodes each image into 280 soft tokens (`processor_config.json`, `max_soft_tokens` 280, the base model's setting and the released one).
- **Server settings:** one server on one NVIDIA H20 96 GB, bf16, `--mem-fraction-static 0.85`, the release `serve_knowline.sh`.
- **How the run was made:** a fresh run (`--fresh`) with the kit at `f5a6e4f` and verification on. The suite was rebuilt with the pinned sources and `vision verify` passed for all 12 sets and 11 benchmarks: Mind2Web labels drawn in Arial Bold, Winoground from the gated `facebook/winoground` after accepting its terms. The latency pass used the same server, one worker, on 30 evenly spaced rows of each benchmark taken from `vision-rows.jsonl`.
- **Training data:** no image training; the model was trained on text only, and its vision tower is the base model's, unchanged. Its only data from a vision suite source is text-only rows built from the Mind2Web **train** split (cleaned-HTML candidate elements and operations, no screenshots): 752 rows (0.08%). The suite's Mind2Web rows come from the `test_website` split, whose websites are not in the train split.
