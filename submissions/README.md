# Model submissions


## Text

| Model | Edition | Decision Index | Complete results | Declared limits |
|---|---|---:|---|---|
| [lg-decision-4B](https://huggingface.co/SunZhengda/lg-decision-4B/tree/607b3f390d542a91665e49196d9d2245ae95f33f) | 0.3 | 78.56 | [scores.json](https://huggingface.co/datasets/SunZhengda/lg-decision-4B-decision-index/blob/5db5f2c28d497e3817a66c6852ed88b7d8add44a/runs/lg-decision-4B/scores.json) | 65,536 tokens per question branch; longer requests are rejected with HTTP 422, not truncated |

- Complete results: https://huggingface.co/datasets/SunZhengda/lg-decision-4B-decision-index/tree/5db5f2c28d497e3817a66c6852ed88b7d8add44a
- Hardware: 1 x NVIDIA H200, bf16


## Run

Install PyTorch 2.10.0 for your CUDA version (the run used `2.10.0+cu129`), then exactly these versions:

```bash
pip install "transformers==5.20.2" "accelerate==1.15.0" "pydantic==2.12.3" \
  "fastapi==0.135.1" "uvicorn==0.41.0" "numpy==1.26.4"
```

Download the model at the pinned revision and start the server. The serving code is in the model repo (`simple-jev/`); the defaults of `start_server.sh` are the settings of this run:

```bash
hf download SunZhengda/lg-decision-4B --revision 607b3f390d542a91665e49196d9d2245ae95f33f --local-dir lg-decision-4B
cd lg-decision-4B

ENV_SERVER_PORT=30000 \
SERVED_MODEL_NAME=lg-decision-4B \
PROMPT_POLICY=baseline \
MAX_MODEL_LEN=65536 \
MAX_BATCH_TOKENS=65536 \
MAX_BATCH_SIZE=16 \
bash simple-jev/start_server.sh
```

Run and score the public suite with the kit (commit `25c6ec7`), from a directory that contains `suite-0.3/`:

```bash
python -m decision_index run --engine http \
  --option base_url=http://127.0.0.1:30000 \
  --option model=lg-decision-4B \
  --compact --out runs/lg-decision-4B
python -m decision_index score --results runs/lg-decision-4B/results.jsonl
```

If a 96 GB card runs out of memory, restart the server with `MAX_BATCH_SIZE=8` (see the PR description).
