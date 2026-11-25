# Model submissions


## Vision

| Model | Edition | Public vision index | Complete results | Image resolution |
|---|---|---:|---|---|
| [Piccolo-0.8B-VL](https://huggingface.co/agentscope-ai/Piccolo-0.8B-VL/tree/3af36a7cc6cc0417f99fb674de331fdded96eee0) | 0.3 | 55.01 | [vision-scores.json](https://huggingface.co/datasets/agentscope-ai/Piccolo_decision_index_vision_results/blob/c0e66cd5b66cdeb18db6d3ce900ce36569ebc39a/Piccolo-0.8B-VL-1124/vision-scores.json) | Capped at 1.6 MP |
| [Piccolo-2B-VL](https://huggingface.co/agentscope-ai/Piccolo-2B-VL/tree/b584e20098788ec8ab86331d95dea984a02116e5) | 0.3 | 66.26 | [vision-scores.json](https://huggingface.co/datasets/agentscope-ai/Piccolo_decision_index_vision_results/blob/c0e66cd5b66cdeb18db6d3ce900ce36569ebc39a/Piccolo-2B-VL-1124/vision-scores.json) | Capped at 1.6 MP |
| [Piccolo-4B-VL](https://huggingface.co/agentscope-ai/Piccolo-4B-VL/tree/7cd5d7a00c355700ce5d32a6f2d7cc9aac50f3f0) | 0.3 | 72.64 | [vision-scores.json](https://huggingface.co/datasets/agentscope-ai/Piccolo_decision_index_vision_results/blob/c0e66cd5b66cdeb18db6d3ce900ce36569ebc39a/Piccolo-4B-VL-1124/vision-scores.json) | Capped at 1.6 MP |
| [Piccolo-9B-VL](https://huggingface.co/agentscope-ai/Piccolo-9B-VL/tree/39cbf3f5f7e18ceade9a36455db985627415807f) | 0.3 | 73.19 | [vision-scores.json](https://huggingface.co/datasets/agentscope-ai/Piccolo_decision_index_vision_results/blob/c0e66cd5b66cdeb18db6d3ce900ce36569ebc39a/Piccolo-9B-VL-1124/vision-scores.json) | Capped at 1.6 MP |
| [Piccolo-27B-VL](https://huggingface.co/agentscope-ai/Piccolo-27B-VL-1124/tree/58af0c6e867f8349414f01116cf4c1df91c01815) | 0.3 | 81.07 | [vision-scores.json](https://huggingface.co/datasets/agentscope-ai/Piccolo_decision_index_vision_results/blob/c0e66cd5b66cdeb18db6d3ce900ce36569ebc39a/Piccolo-27B-VL-1124/vision-scores.json) | Capped at 1.6 MP |

- Complete results: https://huggingface.co/datasets/agentscope-ai/Piccolo_decision_index_vision_results/tree/c0e66cd5b66cdeb18db6d3ce900ce36569ebc39a/Piccolo-0.8B-VL-1124
- Complete results: https://huggingface.co/datasets/agentscope-ai/Piccolo_decision_index_vision_results/tree/c0e66cd5b66cdeb18db6d3ce900ce36569ebc39a/Piccolo-2B-VL-1124
- Complete results: https://huggingface.co/datasets/agentscope-ai/Piccolo_decision_index_vision_results/tree/c0e66cd5b66cdeb18db6d3ce900ce36569ebc39a/Piccolo-4B-VL-1124
- Complete results: https://huggingface.co/datasets/agentscope-ai/Piccolo_decision_index_vision_results/tree/c0e66cd5b66cdeb18db6d3ce900ce36569ebc39a/Piccolo-9B-VL-1124
- Complete results: https://huggingface.co/datasets/agentscope-ai/Piccolo_decision_index_vision_results/tree/c0e66cd5b66cdeb18db6d3ce900ce36569ebc39a/Piccolo-27B-VL-1124
- Hardware: 1 x NVIDIA H20 GPU, bf16
- Declared context limit: 32,768 tokens.
- OCR: enabled, one worker on the same GPU.
- Loop inference: fixed profile for each model size.
- The 2B and 4B checkpoints are unchanged; this submission includes fresh complete evaluation records.


## Run

To run, first clone the code and create separate inference and OCR environments:

```bash
git clone https://github.com/agentscope-ai/Piccolo.git
cd Piccolo
git checkout 39450cf7a954a4af32cb8ddd4a31f1075091cfdc
cd inference/piccolo_inference_sglang
export PLUGIN_ROOT="$PWD"
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install --python .venv/bin/python --prerelease=allow '.[inference]'
uv pip install --python .venv/bin/python 'ninja==1.13.2'
uv venv --python 3.12 .venv-ocr
uv pip install --python .venv-ocr/bin/python --prerelease=allow '.[ocr]' \
  --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/ \
  --index-strategy unsafe-best-match
export PYTHONPATH="$PLUGIN_ROOT"
.venv-ocr/bin/python -m piccolo_inference_sglang.prepare_ocr_models \
  --out "$PWD/ocr_models"
```

Download the model from Hugging Face:

```bash
hf download "agentscope-ai/Piccolo-0.8B-VL" \
  --revision "3af36a7cc6cc0417f99fb674de331fdded96eee0" \
  --local-dir /models/piccolo-0.8B-VL-1124

hf download "agentscope-ai/Piccolo-2B-VL" \
  --revision "b584e20098788ec8ab86331d95dea984a02116e5" \
  --local-dir /models/piccolo-2B-VL-1124

hf download "agentscope-ai/Piccolo-4B-VL" \
  --revision "7cd5d7a00c355700ce5d32a6f2d7cc9aac50f3f0" \
  --local-dir /models/piccolo-4B-VL-1124

hf download "agentscope-ai/Piccolo-9B-VL" \
  --revision "39cbf3f5f7e18ceade9a36455db985627415807f" \
  --local-dir /models/piccolo-9B-VL-1124

hf download "agentscope-ai/Piccolo-27B-VL-1124" \
  --revision "58af0c6e867f8349414f01116cf4c1df91c01815" \
  --local-dir /models/piccolo-27B-VL-1124
```

For the vision set, run
```bash
# start the model server
export PATH="$PLUGIN_ROOT/.venv/bin:/usr/local/cuda/bin:$PATH"
export PYTHONPATH="$PLUGIN_ROOT"
export DECISION_INDEX_API_KEY=EMPTY

"$PLUGIN_ROOT/.venv/bin/python" -B -u \
  -m piccolo_inference_sglang.launch \
  --model /models/piccolo-4B-VL-1124 --size 4B \
  --gpu 0 --port 18728 --host 0.0.0.0 \
  --attention-backend fa3 --memory-fraction 0.75 --context-length 32768 \
  --ocr-python "$PLUGIN_ROOT/.venv-ocr/bin/python" \
  --ocr-model-dir "$PLUGIN_ROOT/ocr_models"

# run the decision index
PLUGIN_ROOT=/path/to/Piccolo/inference/piccolo_inference_sglang
DI_CODE=/path/to/decision-index
DI_SUITE=/path/to/work/vision
RESULT_DIR=/path/to/results/vision-bf16/Piccolo-4B-VL-1124
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
  --option require_ocr=true \
  --workers 1 --chunk 1 \
  --out "$RESULT_DIR" \
  > "$LOG_DIR/Piccolo-4B-VL-1124.log" 2>&1
```
