# Model submissions


## Vision

| Model | Edition | Public vision index | Complete results | Image resolution |
|---|---|---:|---|---|
| [Piccolo-0.8B-VL](https://huggingface.co/agentscope-ai/Piccolo-0.8B-VL/tree/448d1c627793cfb890cc34bbbfd8184cb7ef631d) | 0.3 | 55.10 | [vision-scores.json](https://huggingface.co/datasets/agentscope-ai/Piccolo_decision_index_vision_results/blob/2e85fe9c3e8052271652cdc351dcebee4377eec3/Piccolo-0.8B-VL-1118/vision-scores.json) | Capped at 1.6 MP |
| [Piccolo-2B-VL](https://huggingface.co/agentscope-ai/Piccolo-2B-VL/tree/b584e20098788ec8ab86331d95dea984a02116e5) | 0.3 | 66.26 | [vision-scores.json](https://huggingface.co/datasets/agentscope-ai/Piccolo_decision_index_vision_results/blob/2e85fe9c3e8052271652cdc351dcebee4377eec3/Piccolo-2B-VL-1118/vision-scores.json) | Capped at 1.6 MP |
| [Piccolo-4B-VL](https://huggingface.co/agentscope-ai/Piccolo-4B-VL/tree/08fced0701842c1cb208083b6a412e101c59f46b) | 0.3 | 72.64 | [vision-scores.json](https://huggingface.co/datasets/agentscope-ai/Piccolo_decision_index_vision_results/blob/2e85fe9c3e8052271652cdc351dcebee4377eec3/Piccolo-4B-VL-1118/vision-scores.json) | Capped at 1.6 MP |
| [Piccolo-9B-VL](https://huggingface.co/agentscope-ai/Piccolo-9B-VL/tree/e1ec1d26f35b7338eedfdf91d1935de74435216a) | 0.3 | 73.16 | [vision-scores.json](https://huggingface.co/datasets/agentscope-ai/Piccolo_decision_index_vision_results/blob/2e85fe9c3e8052271652cdc351dcebee4377eec3/Piccolo-9B-VL-1118/vision-scores.json) | Capped at 1.6 MP |
| [Piccolo-27B-VL](https://huggingface.co/agentscope-ai/Piccolo-27B-VL-1118/tree/cd645902b26a58e3b329c8af3cc0730061d3eb4d) | 0.3 | 80.54 | [vision-scores.json](https://huggingface.co/datasets/agentscope-ai/Piccolo_decision_index_vision_results/blob/2e85fe9c3e8052271652cdc351dcebee4377eec3/Piccolo-27B-VL-1118/vision-scores.json) | Capped at 1.6 MP |

- Complete results: https://huggingface.co/datasets/agentscope-ai/Piccolo_decision_index_vision_results/tree/2e85fe9c3e8052271652cdc351dcebee4377eec3/Piccolo-0.8B-VL-1118
- Complete results: https://huggingface.co/datasets/agentscope-ai/Piccolo_decision_index_vision_results/tree/2e85fe9c3e8052271652cdc351dcebee4377eec3/Piccolo-2B-VL-1118
- Complete results: https://huggingface.co/datasets/agentscope-ai/Piccolo_decision_index_vision_results/tree/2e85fe9c3e8052271652cdc351dcebee4377eec3/Piccolo-4B-VL-1118
- Complete results: https://huggingface.co/datasets/agentscope-ai/Piccolo_decision_index_vision_results/tree/2e85fe9c3e8052271652cdc351dcebee4377eec3/Piccolo-9B-VL-1118
- Complete results: https://huggingface.co/datasets/agentscope-ai/Piccolo_decision_index_vision_results/tree/2e85fe9c3e8052271652cdc351dcebee4377eec3/Piccolo-27B-VL-1118
- Hardware: 1 x NVIDIA H20 GPU, bf16
- Declared context limit: 32,768 tokens.
- OCR: enabled, one worker on the same GPU.
- Loop inference: fixed profile for each model size.
- Results retain five initial startup errors per model; all 12,210 unique requests succeeded after resuming.


## Run

To run, first clone the code and create separate inference and OCR environments:

```bash
git clone https://github.com/agentscope-ai/Piccolo.git
cd Piccolo
git checkout 6d0ae9c5a6bbd964a9776998a66f250a4158c957
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
  --revision "448d1c627793cfb890cc34bbbfd8184cb7ef631d" \
  --local-dir /models/piccolo-0.8B-VL-1118

hf download "agentscope-ai/Piccolo-2B-VL" \
  --revision "b584e20098788ec8ab86331d95dea984a02116e5" \
  --local-dir /models/piccolo-2B-VL-1118

hf download "agentscope-ai/Piccolo-4B-VL" \
  --revision "08fced0701842c1cb208083b6a412e101c59f46b" \
  --local-dir /models/piccolo-4B-VL-1118

hf download "agentscope-ai/Piccolo-9B-VL" \
  --revision "e1ec1d26f35b7338eedfdf91d1935de74435216a" \
  --local-dir /models/piccolo-9B-VL-1118

hf download "agentscope-ai/Piccolo-27B-VL-1118" \
  --revision "cd645902b26a58e3b329c8af3cc0730061d3eb4d" \
  --local-dir /models/piccolo-27B-VL-1118
```

For the vision set, run
```bash
# start the model server
export PATH="$PLUGIN_ROOT/.venv/bin:/usr/local/cuda/bin:$PATH"
export PYTHONPATH="$PLUGIN_ROOT"
export DECISION_INDEX_API_KEY=EMPTY

"$PLUGIN_ROOT/.venv/bin/python" -B -u \
  -m piccolo_inference_sglang.launch \
  --model /models/piccolo-4B-VL-1118 --size 4B \
  --gpu 0 --port 18728 --host 0.0.0.0 \
  --attention-backend fa3 --memory-fraction 0.75 --context-length 32768 \
  --ocr-python "$PLUGIN_ROOT/.venv-ocr/bin/python" \
  --ocr-model-dir "$PLUGIN_ROOT/ocr_models"

# run the decision index
PLUGIN_ROOT=/path/to/Piccolo/inference/piccolo_inference_sglang
DI_CODE=/path/to/decision-index
DI_SUITE=/path/to/work/vision
RESULT_DIR=/path/to/results/vision-bf16/Piccolo-4B-VL-1118
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
  > "$LOG_DIR/Piccolo-4B-VL-1118.log" 2>&1
```

