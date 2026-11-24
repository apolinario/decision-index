# Model submissions


## Text

| Model | Edition | Decision Index | Complete results | Declared limits |
|---|---|---:|---|---|
| [lg-decision-27B](https://huggingface.co/SunZhengda/lg-decision-27B/tree/105b19ba1dc824dd2018fadda752be6893647fe3) | 0.3 | 82.83 | [scores.json](https://huggingface.co/datasets/SunZhengda/lg-decision-27B-decision-index/blob/c5819a84dbd9b0ef46bee6557da60c280fac3863/runs/lg-decision-27B/scores.json) | 32,768 tokens per question branch; longer requests are rejected with HTTP 422, not truncated |

- Complete results: https://huggingface.co/datasets/SunZhengda/lg-decision-27B-decision-index/tree/c5819a84dbd9b0ef46bee6557da60c280fac3863
- Hardware: 1 x NVIDIA H200, bf16


## Run

Install PyTorch 2.10.0 for your CUDA version (the run used `2.10.0+cu129`), then exactly these versions:

```bash
pip install "transformers==5.20.2" "accelerate==1.15.0" "pydantic==2.12.3" \
  "fastapi==0.135.1" "uvicorn==0.41.0" "numpy==1.26.4"
```

Download the model at the pinned revision and start the server. The serving code is in the model repo (`simple-jev/`); the defaults of `start_server.sh` are the settings of this run:

```bash
hf download SunZhengda/lg-decision-27B --revision 105b19ba1dc824dd2018fadda752be6893647fe3 --local-dir lg-decision-27B
cd lg-decision-27B

ENV_SERVER_PORT=30000 \
SERVED_MODEL_NAME=lg-decision-27B \
PROMPT_POLICY=baseline \
MAX_MODEL_LEN=32768 \
MAX_BATCH_TOKENS=32768 \
MAX_BATCH_SIZE=4 \
bash simple-jev/start_server.sh
```

Run and score the public suite with the kit (commit `25c6ec7`), from a directory that contains `suite-0.3/`:

```bash
python -m decision_index run --engine http \
  --option base_url=http://127.0.0.1:30000 \
  --option model=lg-decision-27B \
  --compact --out runs/lg-decision-27B
python -m decision_index score --results runs/lg-decision-27B/results.jsonl
```

If a 96 GB card runs out of memory, see the memory notes in the PR description.
