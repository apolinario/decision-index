# Standard One 3B SH (v4)

[StandardThinking/StandardOne-3B-SH](https://huggingface.co/StandardThinking/StandardOne-3B-SH) (tag `v4`, commit `7f56f73be882bc3fb2b69abe609bab1de8297d77`) is Standard One 3B (Ministral 3 3B base) fine-tuned with LoRA and merged into bfloat16 weights, plus a float32 joint schema head that reads the final layer and layer 19. Apache-2.0.

- It reads a request once and scores every option of every question from the hidden states in one prefill; nothing is generated.
- Prompt format: the repo's chat template with no default system message (the repo default).

| | |
|---|---|
| Decision Index 0.3 public index | 58.08 (raw 68.88) |
| Requests | 140,178 of 140,178 scoreable, all answered, none unsupported, nothing truncated, no errors |
| Request time in `scores.json` | median 143.8 ms, mean 541.8 ms, p95 783.5 ms. Measured with 12 client shards in parallel against one engine on a GPU shared with production serving, so it includes queueing; not measured on an RTX PRO 6000 |
| Results | [`StandardThinking/StandardOne-SH-decision-index`](https://huggingface.co/datasets/StandardThinking/StandardOne-SH-decision-index/tree/main/runs/standardone-3b-sh-v4) `runs/standardone-3b-sh-v4` |

Areas (skill x100, coverage 100 % in every area): Knowledge & Reasoning 36.26, Language Understanding 57.53, Retrieval & Classification 66.98, Tools & Automation 85.74, Arts & Human Taste 47.47.

## Run

The engine is the Standard One engine, our SGLang fork with the schema-head decision readout
([standard-one-sglang](https://github.com/standardthinkingai/standard-one-sglang) +
[standard-one-adapter](https://github.com/standardthinkingai/standard-one-adapter), Apache-2.0); the model card section
"Fast serving with the Standard One engine" builds the image with one `docker build`. `di_proxy.py` (this folder) sits
between the kit and the engine: on the engine's three capacity refusals (more than 255 or fewer than 2 options, more than
10 score levels, too many tokens) it appends the kit's capacity marker to the 422 message so the kit records them as
`unsupported`; every other request and response passes through unchanged.

```sh
hf download StandardThinking/StandardOne-3B-SH --revision v4 --local-dir StandardOne-3B-SH
docker run -d --gpus all --network host --ipc host --shm-size 32g -v $PWD/StandardOne-3B-SH:/models/m:ro standard-one-engine:cu13 \
  python3 -m sglang.launch_server --model-path /models/m --decision-readout-path /models/m/head \
  --served-model-name standardthinking/standard-one-3b-sh --host 127.0.0.1 --port 30251 --tp-size 1 --model-impl sglang --dtype bfloat16 \
  --attention-backend triton --context-length 131072 --max-running-requests 32 --mem-fraction-static 0.3 \
  --chunked-prefill-size -1 --model-config-parser hf --load-format safetensors --disable-radix-cache \
  --disable-decode-cuda-graph --disable-prefill-cuda-graph --decision-deadline-seconds 600
python di_proxy.py --engine http://127.0.0.1:30251 --port 30261     # needs httpx
python -m decision_index pipeline --engine http --option base_url=http://127.0.0.1:30261 \
  --option model=standardthinking/standard-one-3b-sh --out runs/standardone-3b-sh-v4
```

Our run used the ROCm build of the same engine (image `st-sglang:decision-qwen35-776e539961-60dac5b`) on one AMD Instinct MI350X
(`--device /dev/kfd --device /dev/dri --group-add video -e SGLANG_USE_AITER=0` instead of `--gpus all`, 60 GB of the GPU's memory) that
was also serving production traffic, and the kit at `9eb2dbe`. The 0.3 rows were split round-robin into 24 shards run with
`run --rows <shard>` by 12 kit processes in parallel against the one engine; the shard results were concatenated and scored once with
`score --edition 0.3`. Two rows outside the scoreable set (BRIGHT `stackoverflow:21:chunk0` and `stackoverflow:35:chunk0`) got a 422
from the engine without a capacity marker in both passes and stay recorded as `error`; they do not count toward the index. Our harness
labels every engine `standardthinking/standard-one-8b` in `--served-model-name`, so that string appears in `results.jsonl` and
`environments/`; the loaded weights were the v4 3B files. The results README describes the run step by step and holds the engine command.

## Settings and limits

- bfloat16 backbone, float32 head, no default system message.
- Context 131,072 tokens for this run. The model card serves 32,768; we raised it so no request was refused for length.
  The model was trained on inputs up to 32k.
- Declared limits: 2-255 options per choice, 2-10 score levels, 131,072 tokens. Nothing in the 140,178 scoreable requests was
  unsupported or truncated (a few dozen rows outside the scoreable set, ToolRet requests beyond the declared limits, are
  recorded as `unsupported`).

## Training data

- v4 continues from v3 with one more round of training on decision data, mostly our own synthetic items, with earlier data
  replayed. The round also fixes answer-position and label-rate drift seen in the previous checkpoint.
- Every training row of this round was screened against the 0.3 public rows (exact state match and shared rare 13-word passages);
  flagged rows were removed.
- Compared with v3 (skill x100): PhishNChips 57.9 → 69.5, RAGTruth 21.9 → 50.6, GSM8K 17.2 → 41.2, FinEntity 79.9 → 82.9.
  Down: POP909-CL 78.3 → 67.2, HoVer 63.9 → 60.1, GPQA Diamond 10.9 → 6.1.
- Internal and new-domain checks sit slightly below the previous internal checkpoint: our own held-out new-domain set scored
  65.9 against 67.1 for that checkpoint.

Earlier versions were trained on, among other data, our own request-format conversions of the train splits of public
datasets, some of which are also sources of Decision Index benchmarks (for example When2Call, HellaSwag, ESCI, HoVer, FEVER,
VitaminC, MedMCQA). Only train splits were used. v3 added decision data built from public sources the Decision Index also
draws on, keeping the evaluated items out (the Home appliance simulator's public generator re-seeded, BBH-style tasks from our
own code, When2Call train-split pairs, CLadder questions not in the 0.3 rows, the ContractNLI and Habermas Machine train splits,
POP909 beats at least 8 beats from every evaluated beat), and our own synthetic items in several of the benchmark families; earlier
rounds' data was replayed at a lower weight. The v4 round is the first for which every training row was screened against the 0.3
public rows as described above.
