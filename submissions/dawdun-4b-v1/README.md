# DawDun-4B-v1 Decision Index inference

The deployment includes the inference path from the public evaluation, packaged as a Python engine and a local HTTP server. It loads the pinned DawDun-4B-v1 export with its calibrated 255-class readout. The engine preserves the evaluation's one-question encoding, last-token readout, bidirectional full attention, FP32 projection and softmax after per-type temperature scaling.

A CPU comparison against the archived evaluation engine used the same source weights and runtime for two text and two vision requests. All decisions and probabilities matched exactly on those requests. GPU output parity with the archived scores remains to be verified.

This adapter accepts a compatible 255-class inference export. It formats the question as evidence, instructions, and lettered options, then returns a probability per option. A request may contain several questions. Each question gets its own forward pass. The readout uses the final prompt token and the exported per-type temperatures.

## Model bundle

Use a Transformers-format export directory or a Hugging Face model repository and pinned commit. The directory must contain the Qwen3.5 backbone and processor, chat template, tokenizer, `classification_config.json` with the option tokens and per-type temperatures, and `readout.safetensors`. The readout tensor is named `weight` and has shape `[255, 2560]`. Keep these files together. A training checkpoint can first be converted to an inference export by its owner.

The evaluation used NVIDIA H800 workers with one engine per device, Python 3.12, PyTorch 2.6.0 CUDA 12.4, Transformers 5.20.1, Triton 3.2.0, FLA Core 0.5.2, and causal-conv1d 1.5.0.post8. For GPU performance testing, install FLA and causal-conv1d builds compatible with your PyTorch, CUDA and GPU. Transformers can use slower PyTorch implementations when those kernels are unavailable. Use an environment without the CUDA-only extensions for a CPU smoke test.

## Install and start the server

From the repository root, enter `submissions/dawdun-4b-v1` in a Python 3.12 environment. Install the PyTorch and torchvision pair for your GPU. The archived H800 runs used CUDA 12.4. For the RTX PRO 6000, the CUDA 12.8 build supports Blackwell. [PyTorch GPU support](https://pytorch.org/blog/pytorch-2-7/) · [PyTorch wheel versions](https://pytorch.org/get-started/previous-versions/#v271)

```bash
# RTX PRO 6000
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1 torchvision==0.22.1
```

```bash
# H800
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0
```

Then install the remaining dependencies and start the server.

```bash
python -m pip install -r requirements.txt

MODEL=Doffe/DawDun-4B-v1
MODEL_COMMIT=157143ee296358abb9557e4bf04c2e2d15a7951d
python -m head255_di.server --model "$MODEL" --revision "$MODEL_COMMIT" \
  --device cuda --host 127.0.0.1 --port 8000
```

The server downloads the pinned model on first launch and uses GPU 0 as visible to the process. Use `CUDA_VISIBLE_DEVICES` to select a GPU. For an existing local export, pass its directory to `--model` and omit `--revision`. The CUDA path uses a BF16 backbone and FP32 readout. The calibrated Choice, NOUL and Score temperatures come from the model bundle.

- `POST /v1/systemone` takes `state` and a mapping of typed `questions`. It returns `answers` keyed by the original question IDs and `usage.input_tokens`.
- `POST /v1/decisions` accepts the same structured request with an ordered `images` array of image data URLs or raw base64. It also returns the processed `image_pixels` per image and the `image_tokens`.

```bash
curl -fsS http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"Doffe/DawDun-4B-v1","state":"The sign is red.","questions":{"color":{"type":"choice","instructions":"What color is the sign?","criteria":{"red":"Red","blue":"Blue"}}}}'
```

## Run with the official kit

Set `TEXT_SUITE` and `VISION_SUITE` to the frozen local suite directories for the respective commands. The text runner can drive the server with the kit's `http` engine.

```bash
python -m decision_index pipeline --edition 0.3 --suite-dir "$TEXT_SUITE" \
  --engine http --option base_url=http://127.0.0.1:8000 \
  --model Doffe/DawDun-4B-v1 --compact --out runs/dawdun-4b-v1-text-http
```

The kit's `decide` client can exercise the vision endpoint with `--option protocol=decisions` and `--option base_url=http://127.0.0.1:8000`. It applies its own image resize by default. Use the in-process vision command below for processor-only resizing and the processed pixel counts used by the archived evaluation. The same Python engine also provides an in-process text entry point.

```bash
MODEL=Doffe/DawDun-4B-v1
MODEL_COMMIT=157143ee296358abb9557e4bf04c2e2d15a7951d

python -m decision_index pipeline --edition 0.3 \
  --suite-dir "$TEXT_SUITE" --engine head255_di.engine:Head255Engine \
  --model "$MODEL" --option revision="$MODEL_COMMIT" --option context_tokens=262144 --compact \
  --out runs/dawdun-4b-v1-text

python -m decision_index vision pipeline --dir "$VISION_SUITE" \
  --engine head255_di.engine:Head255Engine --model "$MODEL" --option revision="$MODEL_COMMIT" \
  --option context_tokens=262144 --option max_pixels=1638400 \
  --workers 1 --compact --out runs/dawdun-4b-v1-vision
```

Use a fresh output directory when changing the model or an inference setting. The default context limit is 262,144 tokens, with no truncation. The default processed image bounds are 65,536–1,638,400 pixels. The public results record the archived evaluation runs. Decision Index maintainers run the private tests that contribute to the Full Score. [Decision Index submission requirements](https://github.com/apolinario/decision-index#submitting-a-model-to-the-leaderboard)
