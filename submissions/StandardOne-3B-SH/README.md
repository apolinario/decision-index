# Standard One 3B SH (v3)

[StandardThinking/StandardOne-3B-SH](https://huggingface.co/StandardThinking/StandardOne-3B-SH) (tag `v3`, commit `c6c6af830244daccb372d41301a26a628d91b08a`) is Standard One 3B (Ministral 3 3B base) fine-tuned with LoRA and merged into bfloat16 weights, plus a float32 joint schema head that reads the final layer and layer 19. Apache-2.0.

- It reads a request once and scores every option of every question from the hidden states in one prefill; nothing is generated.
- Prompt format: the repo's chat template with no default system message (the repo default).

| | |
|---|---|
| Decision Index 0.3 public index | 51.53 (raw 64.29) |
| Requests | 140,178 of 140,178 scoreable, all answered, none unsupported, nothing truncated, no errors |
| Request time in `scores.json` | median 313.7 ms, mean 1,144.0 ms. Measured with many client shards in parallel per engine (some engines on a GPU shared with other serving), so it includes queueing; not measured on an RTX PRO 6000 |
| Results | [`StandardThinking/StandardOne-SH-decision-index`](https://huggingface.co/datasets/StandardThinking/StandardOne-SH-decision-index/tree/main/runs/standardone-3b-sh-v3) `runs/standardone-3b-sh-v3` |

Areas (skill x100, coverage 100 % in every area): Knowledge & Reasoning 30.30, Language Understanding 49.46, Retrieval & Classification 63.30, Tools & Automation 79.83, Arts & Human Taste 36.50.

## Run

The engine is the Standard One engine, our SGLang fork with the schema-head decision readout
([standard-one-sglang](https://github.com/standardthinkingai/standard-one-sglang) +
[standard-one-adapter](https://github.com/standardthinkingai/standard-one-adapter), Apache-2.0); the model card section
"Fast serving with the Standard One engine" builds the image with one `docker build`. `di_proxy.py` (this folder) sits
between the kit and the engine: on the engine's three capacity refusals (more than 255 or fewer than 2 options, more than
10 score levels, too many tokens) it appends the kit's capacity marker to the 422 message so the kit records them as
`unsupported`; every other request and response passes through unchanged.

```sh
hf download StandardThinking/StandardOne-3B-SH --revision v3 --local-dir StandardOne-3B-SH
docker run -d --gpus all --network host --ipc host --shm-size 32g -v $PWD/StandardOne-3B-SH:/models/m:ro standard-one-engine:cu13 \
  python3 -m sglang.launch_server --model-path /models/m --decision-readout-path /models/m/head \
  --served-model-name standardthinking/standard-one-3b-sh --host 127.0.0.1 --port 30251 --tp-size 1 --model-impl sglang --dtype bfloat16 \
  --attention-backend triton --context-length 131072 --max-running-requests 32 --mem-fraction-static 0.3 \
  --chunked-prefill-size -1 --model-config-parser hf --load-format safetensors --disable-radix-cache \
  --disable-decode-cuda-graph --disable-prefill-cuda-graph --decision-deadline-seconds 600
python di_proxy.py --engine http://127.0.0.1:30251 --port 30261     # needs httpx
python -m decision_index pipeline --engine http --option base_url=http://127.0.0.1:30261 \
  --option model=standardthinking/standard-one-3b-sh --out runs/standardone-3b-sh-v3
```

Our run used the ROCm build of the same engine on AMD Instinct MI350X (`--device /dev/kfd --device /dev/dri
--group-add video -e SGLANG_USE_AITER=0` instead of `--gpus all`) and the kit at `9eb2dbe`. To finish sooner, the 0.3 rows were
split round-robin into 48 shards run with `run --rows <shard>` against one engine per host on up to four MI350X hosts; unfinished
shards were moved between hosts with their partial results, which the kit resumes from. Before scoring, a script rebuilt the
shard assignment from the suite and checked that every row is present once, in its own shard; then the shard results were
concatenated and scored once with `score --edition 0.3`. The results README describes the run step by step and holds every
engine command and script.

## Settings and limits

- bfloat16 backbone, float32 head, no default system message.
- Context 131,072 tokens for this run. The model card serves 32,768; we raised it so no request was refused for length.
  The model was trained on inputs up to 32k.
- Declared limits: 2-255 options per choice, 2-10 score levels, 131,072 tokens. Nothing in the 140,178 scoreable requests was
  unsupported or truncated (a few dozen rows outside the scoreable set, ToolRet requests beyond the declared limits, are
  recorded as `unsupported`).

## Training data

Earlier versions were trained on, among other data, our own request-format conversions of the train splits of public
datasets, some of which are also sources of Decision Index benchmarks (for example When2Call, HellaSwag, ESCI, HoVer, FEVER,
VitaminC, MedMCQA). Only train splits were used.

This version adds two rounds of further training:

- Decision data built from public sources that the Decision Index also draws on, keeping the evaluated items out: the
  Home appliance simulator's own public generator re-seeded with new households, devices and wording (the evaluated cases
  were not used); BBH-style tasks generated by our own code; When2Call train-split preference pairs turned into
  multiple-choice; CLadder questions that are not in the 0.3 rows; the ContractNLI and Habermas Machine train splits;
  POP909 beats at least 8 beats away from every evaluated beat.
- Our own synthetic decision items, written by LLMs in varied styles and topics and kept only when code or a second model
  confirmed the gold answer: stance toward a topic, unsupported content in grounded answers, phishing emails, sarcasm,
  where a person believes an object is in a story, whether to call a tool / ask / decline, and device requests under
  interacting household or facility rules. No text from the Decision Index rows was used to write them.

Earlier rounds' data was replayed at a lower weight. We did not run an item-level overlap check against the 0.3 suite rows.
