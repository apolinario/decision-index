# Standard One 3B SH (v2)

[StandardThinking/StandardOne-3B-SH](https://huggingface.co/StandardThinking/StandardOne-3B-SH) (tag `v2`, commit `1e94d7af2a71eb0fd9449e290ca725f7d0d430c6`) is Standard One 3B (Ministral 3 3B base) fine-tuned with LoRA and merged into bfloat16 weights, plus a float32 joint schema head that reads the final layer and layer 19. Apache-2.0.

- It reads a request once and scores every option of every question from the hidden states in one prefill; nothing is generated.
- Prompt format: the repo's chat template with no default system message (the repo default).

| | |
|---|---|
| Decision Index 0.3 public index | 42.90 (raw 57.27) |
| Requests | 140,178 of 140,178 scoreable, all answered, none unsupported, nothing truncated, no errors |
| Request time in `scores.json` | median 214.7 ms, mean 1,026.6 ms. Measured with 24 client shards in parallel against one engine on a GPU shared with other serving, so it includes queueing; not measured on an RTX PRO 6000 |
| Results | [`StandardThinking/StandardOne-SH-decision-index`](https://huggingface.co/datasets/StandardThinking/StandardOne-SH-decision-index/tree/main/runs/standardone-3b-sh-v2) `runs/standardone-3b-sh-v2` |

Areas (skill x100, coverage 100 % in every area): Knowledge & Reasoning 25.25, Language Understanding 44.19, Retrieval & Classification 59.66, Tools & Automation 60.33, Arts & Human Taste 19.77.

## Run

The engine is the Standard One engine, our SGLang fork with the schema-head decision readout
([standard-one-sglang](https://github.com/standardthinkingai/standard-one-sglang) +
[standard-one-adapter](https://github.com/standardthinkingai/standard-one-adapter), Apache-2.0); the model card section
"Fast serving with the Standard One engine" builds the image with one `docker build`. `di_proxy.py` (this folder) sits
between the kit and the engine: on the engine's three capacity refusals (more than 255 or fewer than 2 options, more than
10 score levels, too many tokens) it appends the kit's capacity marker to the 422 message so the kit records them as
`unsupported`; every other request and response passes through unchanged.

```sh
hf download StandardThinking/StandardOne-3B-SH --revision v2 --local-dir StandardOne-3B-SH
docker run -d --gpus all --network host --ipc host --shm-size 32g -v $PWD/StandardOne-3B-SH:/models/m:ro standard-one-engine:cu13 \
  python3 -m sglang.launch_server --model-path /models/m --decision-readout-path /models/m/head \
  --served-model-name standardthinking/standard-one-3b-sh --host 127.0.0.1 --port 30251 --tp-size 1 --model-impl sglang --dtype bfloat16 \
  --attention-backend triton --context-length 131072 --max-running-requests 32 --mem-fraction-static 0.3 \
  --chunked-prefill-size -1 --model-config-parser hf --load-format safetensors --disable-radix-cache \
  --disable-decode-cuda-graph --disable-prefill-cuda-graph --decision-deadline-seconds 600
python di_proxy.py --engine http://127.0.0.1:30251 --port 30261     # needs httpx
python -m decision_index pipeline --engine http --option base_url=http://127.0.0.1:30261 \
  --option model=standardthinking/standard-one-3b-sh --out runs/standardone-3b-sh-v2
```

Our run used the ROCm build of the same engine on one AMD Instinct MI350X (`--device /dev/kfd --device /dev/dri
--group-add video -e SGLANG_USE_AITER=0` instead of `--gpus all`), the kit at `9eb2dbe`, the 0.3 rows split round-robin
into 24 shards run in parallel with `run --rows <shard>` against that one engine, one resume pass for errors, then the
shard results concatenated and scored with `score --edition 0.3`. The exact engine command, the shard script and the run
script are in the results directory.

## Settings and limits

- bfloat16 backbone, float32 head, no default system message.
- Context 131,072 tokens for this run. The model card serves 32,768; we raised it so no request was refused for length.
  The model was trained on inputs up to 32k.
- Declared limits: 2-255 options per choice, 2-10 score levels, 131,072 tokens. Nothing in the 0.3 rows was unsupported
  or truncated.

## Training data

The training data includes our own request-format conversions of the train splits of public datasets, some of which are
also sources of Decision Index benchmarks (for example When2Call, HellaSwag, ESCI, HoVer, FEVER, VitaminC, MedMCQA). Only
train splits were used; we did not run an item-level overlap check against the 0.3 suite rows.
