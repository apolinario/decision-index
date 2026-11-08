# Model submissions


## Vision

| Model | Edition | Public vision index | Complete results | Image resolution |
|---|---|---:|---|---|
| [Piccolo-4B-VL](https://huggingface.co/agentscope-ai/Piccolo-4B-VL/tree/a8fb097a5aacee7bdfc0ae4b511dd43e8ce351c4) | 0.3 | 71.55 | [vision-scores.json](https://huggingface.co/datasets/agentscope-ai/Piccolo_decision_index_vision_results/blob/fcbbafcb6f27f8d2ef228db2b7b57eafe76ad024/Piccolo-4B-VL-1107/vision-scores.json) | Capped at 1.6 MP |
| [Piccolo-9B-VL](https://huggingface.co/agentscope-ai/Piccolo-9B-VL/tree/49fc36378f9bef27999015cccd83e92664f0ea92) | 0.3 | 71.76 | [vision-scores.json](https://huggingface.co/datasets/agentscope-ai/Piccolo_decision_index_vision_results/blob/fcbbafcb6f27f8d2ef228db2b7b57eafe76ad024/Piccolo-9B-VL-1107/vision-scores.json) | Capped at 1.6 MP |
| [Piccolo-27B-VL](https://huggingface.co/agentscope-ai/Piccolo-27B-VL-1108/tree/fb994578f3080a1d1dcc0c4f2977f7169add5e98) | 0.3 | 79.97 | [vision-scores.json](https://huggingface.co/datasets/agentscope-ai/Piccolo_decision_index_vision_results/blob/fcbbafcb6f27f8d2ef228db2b7b57eafe76ad024/Piccolo-27B-VL-1107/vision-scores.json) | Capped at 1.6 MP |

- Complete results: https://huggingface.co/datasets/agentscope-ai/Piccolo_decision_index_vision_results/tree/fcbbafcb6f27f8d2ef228db2b7b57eafe76ad024/Piccolo-4B-VL-1107
- Complete results: https://huggingface.co/datasets/agentscope-ai/Piccolo_decision_index_vision_results/tree/fcbbafcb6f27f8d2ef228db2b7b57eafe76ad024/Piccolo-9B-VL-1107
- Complete results: https://huggingface.co/datasets/agentscope-ai/Piccolo_decision_index_vision_results/tree/fcbbafcb6f27f8d2ef228db2b7b57eafe76ad024/Piccolo-27B-VL-1107
- Hardware: 1 x NVIDIA H20 GPU, bf16
- Declared context limit: 131,072 tokens.
- OCR: enabled, one worker on the same GPU.


## Run

To run, first clone the code and create a virtual environment:

```bash
git clone https://github.com/agentscope-ai/Piccolo.git
cd Piccolo
git checkout bb5d912f22855e3b6bdb3d0319164874197d1d1a
cd inference/piccolo_inference_sglang
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install --prerelease=allow .
uv pip install --python .venv/bin/python --prerelease=allow '.[ocr]' \
  --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/ \
  --index-strategy unsafe-best-match
uv run --no-sync python -m piccolo_inference_sglang.prepare_ocr_models \
  --out "$PWD/ocr_models"
```

Download the model from Hugging Face:

```bash
hf download "agentscope-ai/Piccolo-4B-VL" \
  --revision "a8fb097a5aacee7bdfc0ae4b511dd43e8ce351c4" \
  --local-dir /models/piccolo-4B-VL-1107

hf download "agentscope-ai/Piccolo-9B-VL" \
  --revision "49fc36378f9bef27999015cccd83e92664f0ea92" \
  --local-dir /models/piccolo-9B-VL-1107

hf download "agentscope-ai/Piccolo-27B-VL-1108" \
  --revision "fb994578f3080a1d1dcc0c4f2977f7169add5e98" \
  --local-dir /models/piccolo-27B-VL-1107
```

For the vision set, run
```bash
# start the model server
PICCOLO_OCR_ENABLED=1 \
PICCOLO_OCR_PYTHON="$PWD/.venv/bin/python" \
PICCOLO_OCR_MODEL_DIR="$PWD/ocr_models" \
PYTHON="$PWD/.venv/bin/python" \
  bash scripts/launch_vision.sh /models/piccolo-4B-VL-1107 0 18728 \
  --context-length 131072 --max-total-tokens 131072

# run the decision index
PLUGIN_ROOT=/path/to/Piccolo/inference/piccolo_inference_sglang
DI_CODE=/path/to/decision-index
DI_SUITE=/path/to/work/vision
RESULT_DIR=/path/to/results/vision-bf16/Piccolo-4B-VL-1107
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
  > "$LOG_DIR/Piccolo-4B-VL-1107.log" 2>&1
```

