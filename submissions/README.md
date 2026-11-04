# Model submissions


## Vision

| Model | Edition | Public vision index | Complete results | Image resolution |
|---|---|---:|---|---|
| [Piccolo-27B-VL](https://huggingface.co/agentscope-ai/Piccolo-27B-VL/tree/a0080cdaa77c9a28cd27e9e6a43c64e279b29a27) | 0.3 | 77.03 | [vision-scores.json](https://huggingface.co/datasets/agentscope-ai/Piccolo_decision_index_vision_results/blob/10850c65b8873076dd809f9848632dae73e7d1e6/Piccolo-27B-VL-1102/vision-scores.json) | Capped at 1.6 MP |

- Complete results: https://huggingface.co/datasets/agentscope-ai/Piccolo_decision_index_vision_results/tree/10850c65b8873076dd809f9848632dae73e7d1e6/Piccolo-27B-VL-1102
- Hardware: 1 x NVIDIA H20 GPU, bf16
- Declared context limit: 32,768 tokens.


## Run

To run, first clone the code and create a virtual environment:

```bash
git clone https://github.com/agentscope-ai/Piccolo.git
cd Piccolo
git checkout 0b35b631b5e4a9e4a96b059f96ad1c095ffa0253
cd inference/piccolo_inference_sglang
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install --prerelease=allow .
```

Download the model from Hugging Face:

```bash
hf download agentscope-ai/Piccolo-27B-VL \
  --revision a0080cdaa77c9a28cd27e9e6a43c64e279b29a27 \
  --local-dir /models/piccolo-27B-VL-1102
```

For the vision set, run
```bash
# start the model server
bash scripts/launch_vision.sh /models/piccolo-27B-VL-1102 0 18728

# run the decision index
PLUGIN_ROOT=/path/to/Piccolo/inference/piccolo_inference_sglang
DI_CODE=/path/to/decision-index
DI_SUITE=/path/to/work/vision
RESULT_DIR=/path/to/results/vision-bf16/Piccolo-27B-VL-1102
LOG_DIR=/path/to/logs/vision-bf16

export PYTHONPATH="$DI_CODE:$PLUGIN_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export DECISION_INDEX_API_KEY=EMPTY
mkdir -p "$LOG_DIR"
cd "$DI_CODE"

python -u -m decision_index vision pipeline \
  --dir "$DI_SUITE" \
  --engine piccolo_inference_sglang.decision_index_engine:VisionEngine \
  --model Piccolo \
  --option base_url=http://model-host:18728 \
  --workers 1 --chunk 1 \
  --out "$RESULT_DIR" \
  > "$LOG_DIR/Piccolo-27B-VL-1102.log" 2>&1
```
