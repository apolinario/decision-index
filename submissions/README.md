# Model submissions


## Text

| Model | Edition | Decision Index | Complete results | Declared limits |
|---|---|---:|---|---|
| [Piccolo-27B](https://huggingface.co/agentscope-ai/Piccolo-27B/tree/9cab0ecef7372d29ac83b50ac5e49cead2caa173) | 0.3 | 68.08 | [scores.json](https://huggingface.co/datasets/agentscope-ai/Piccolo_decision_index_results/blob/6286b9df2276abbd93fb61e0acb1c2ea366c5faa/Piccolo-27B-1028/scores.json) | Input context: 65,535 tokens |
| [Piccolo-9B](https://huggingface.co/agentscope-ai/Piccolo-9B/tree/3e5e54b1554ad6ad85b46980719d6662a5980fd7) | 0.3 | 60.3 | [scores.json](https://huggingface.co/datasets/agentscope-ai/Piccolo_decision_index_results/blob/6286b9df2276abbd93fb61e0acb1c2ea366c5faa/Piccolo-9B-1028/scores.json) | Input context: 65,535 tokens |
| [Piccolo-4B](https://huggingface.co/agentscope-ai/Piccolo-4B/tree/fe6a7af0c6eaf363612c90051243b4e8aeacc90e) | 0.3 | 57.14 | [scores.json](https://huggingface.co/datasets/agentscope-ai/Piccolo_decision_index_results/blob/6286b9df2276abbd93fb61e0acb1c2ea366c5faa/Piccolo-4B-1028/scores.json) | Input context: 65,535 tokens |

- Complete results: https://huggingface.co/datasets/agentscope-ai/Piccolo_decision_index_results/tree/6286b9df2276abbd93fb61e0acb1c2ea366c5faa
- Hardware: 1 x NVIDIA H20 GPU, fp8

> The submitted model names are Piccolo. The recorded `NaiveVisionEngine` and the published `VisionEngine` refer to the same engine.


## Vision

| Model | Edition | Public vision index | Complete results | Image resolution |
|---|---|---:|---|---|
| [Piccolo-27B](https://huggingface.co/agentscope-ai/Piccolo-27B/tree/9cab0ecef7372d29ac83b50ac5e49cead2caa173) | 0.3 | 74.03 | [vision-scores.json](https://huggingface.co/datasets/agentscope-ai/Piccolo_decision_index_vision_results/blob/76b514b1c0fd24a6d1b5b2411446f0af4772e0fa/Piccolo-27B-1028/vision-scores.json) | Capped at 1.6 MP |
| [Piccolo-9B](https://huggingface.co/agentscope-ai/Piccolo-9B/tree/3e5e54b1554ad6ad85b46980719d6662a5980fd7) | 0.3 | 66.49 | [vision-scores.json](https://huggingface.co/datasets/agentscope-ai/Piccolo_decision_index_vision_results/blob/76b514b1c0fd24a6d1b5b2411446f0af4772e0fa/Piccolo-9B-1028/vision-scores.json) | Capped at 1.6 MP |
| [Piccolo-4B](https://huggingface.co/agentscope-ai/Piccolo-4B/tree/fe6a7af0c6eaf363612c90051243b4e8aeacc90e) | 0.3 | 65.33 | [vision-scores.json](https://huggingface.co/datasets/agentscope-ai/Piccolo_decision_index_vision_results/blob/76b514b1c0fd24a6d1b5b2411446f0af4772e0fa/Piccolo-4B-1028/vision-scores.json) | Capped at 1.6 MP |

- Complete results: https://huggingface.co/datasets/agentscope-ai/Piccolo_decision_index_vision_results/tree/76b514b1c0fd24a6d1b5b2411446f0af4772e0fa
- Hardware: 1 x NVIDIA H20 GPU, bf16
- Declared context limit (all three models): 32,768 tokens.

> The submitted model names are Piccolo. The recorded `NaiveVisionEngine` and the published `VisionEngine` refer to the same engine.

## Run

To run, first clone the code and create a virtual environment
```bash
git clone https://github.com/agentscope-ai/Piccolo.git
cd Piccolo
git checkout 77148adf72abbfb1234c19fd42afa9fa7e9e2ca2
cd inference/piccolo_inference_sglang
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install --prerelease=allow .
```

Download the models from Hugging Face
```bash
hf download agentscope-ai/Piccolo-27B --revision 9cab0ecef7372d29ac83b50ac5e49cead2caa173 --local-dir /models/piccolo-27B-1028
hf download agentscope-ai/Piccolo-9B --revision 3e5e54b1554ad6ad85b46980719d6662a5980fd7 --local-dir /models/piccolo-9B-1028
hf download agentscope-ai/Piccolo-4B --revision fe6a7af0c6eaf363612c90051243b4e8aeacc90e --local-dir /models/piccolo-4B-1028
```

To evaluate the models, for example, the Piccolo-4B, we first start the model server and then run the decision index.

For text set, run
```bash
# start the model server
bash scripts/launch_sglang.sh /models/piccolo-4B-1028 0 18604

# run the decision index
DI_CODE=/path/to/decision-index
DI_SUITE=/path/to/suite-0.3
RESULT_DIR=/path/to/results/text-fp8/piccolo-4B-1028
LOG_DIR=/path/to/logs/text-fp8

export PYTHONPATH="$DI_CODE"
export PYTHONDONTWRITEBYTECODE=1
export DECISION_INDEX_API_KEY=EMPTY
mkdir -p "$LOG_DIR"
cd "$DI_CODE"

python -u -m decision_index pipeline \
  --edition 0.3 --suite-dir "$DI_SUITE" \
  --engine http \
  --option base_url=http://model-host:18604 \
  --option model=piccolo-4B-1028 \
  --out "$RESULT_DIR" \
  > "$LOG_DIR/piccolo-4B-1028.log" 2>&1 < /dev/null &
```

For the vision set, run
```bash
# start the model server
bash scripts/launch_vision.sh /models/piccolo-4B-1028 0 18704

# run the decision index
PLUGIN_ROOT=/path/to/piccolo_inference_sglang
DI_CODE=/path/to/decision-index
DI_SUITE=/path/to/work/vision
RESULT_DIR=/path/to/results/vision-bf16/piccolo-4B-1028
LOG_DIR=/path/to/logs/vision-bf16

export PYTHONPATH="$DI_CODE:$PLUGIN_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export DECISION_INDEX_API_KEY=EMPTY
mkdir -p "$LOG_DIR"
cd "$DI_CODE"

python -u -m decision_index vision pipeline \
  --dir "$DI_SUITE" \
  --engine piccolo_inference_sglang.decision_index_engine:VisionEngine \
  --model piccolo-4B-1028 \
  --option base_url=http://model-host:18704 \
  --workers 1 --chunk 1 \
  --out "$RESULT_DIR" \
  > "$LOG_DIR/piccolo-4B-1028.log" 2>&1 < /dev/null &
```
