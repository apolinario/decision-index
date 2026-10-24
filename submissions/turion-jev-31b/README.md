# Turion Jev 31B

[Turion Jev 31B](https://huggingface.co/turion-ai/turion-jev-31b/tree/9edd5783ed1735cbf9e3e8ae7a546126ae05b0ce) is a decision model by [Turion AI](https://turion.ai), fine-tuned from google/gemma-4-31B-it on the [Turion AI model training platform](https://turion.ai/training/) and released under Apache-2.0. It returns one probability for every option of a question about a state, read in a single forward pass. It does not generate text.

| | Turion Jev 31B |
|---|---:|
| Decision Index 0.3, public index | **66.16** |
| Requests | 140,620: 140,393 `ok` (140,178 scoreable), 227 unsupported, 0 errors |
| Request time, one request at a time (scoring run) | median 116 · mean 288 · p80 352 ms |
| Hardware | 1× NVIDIA RTX PRO 6000 Blackwell Workstation Edition (96 GB) |
| Model | [`turion-ai/turion-jev-31b`](https://huggingface.co/turion-ai/turion-jev-31b/tree/9edd5783ed1735cbf9e3e8ae7a546126ae05b0ce): public |
| Results | [runs/turion-jev-31b](https://huggingface.co/datasets/turion-ai/turion-jev-31b-decision-index-results/tree/cd7b034e9d249ea5556565286234b1393de2fb0c/runs/turion-jev-31b) (compact: no suite text) |

## Engine

`turion_jev.kit_engine:TurionJev`, shipped in the model repo under `code/` (Apache-2.0).
- It runs in-process on vLLM 0.32.0 (torch 2.13.0+cu130, transformers 5.19.0), one request at a time.
- The weights are BF16, quantised online to FP8.
- It lists every question of a request before the state and reads the log-probability of each option's one-token code (up to 255 per question) after the prompt, normalised over the options shown, at the per-type temperatures in `calibration.json`.

Options, exactly as run:

```text
--engine turion_jev.kit_engine:TurionJev
--option model_dir=<local copy of turion-ai/turion-jev-31b> --option backend=vllm --option quantization=fp8
--option order_avg_max=2 --option second_pass_max_questions=1 --option repeat_max_questions=1
--option warm_prefix=true --option recheck=false
--compact
```

- `order_avg_max=2` with `second_pass_max_questions=1`: choice questions with at most 2 options are also scored in reversed order and averaged, in requests with one question.
- `repeat_max_questions=1`: the input (up to 12,000 characters, from `prompt_config.json`) is repeated once, in requests with one question.
- `warm_prefix=true`: a request's shared prefix is scored once before its questions.
- `recheck=false`: no second pass for unsure long choice questions.

vLLM settings set by the engine: BF16 with online FP8, `max_model_len` 65,600, prefix caching, `gpu_memory_utilization` 0.92, `max_logprobs` 255.

One 96 GB GPU holds the model.

## Declared limits

Up to 65,536 prompt tokens per request and 255 options per question. A longer request is refused (`Unsupported`), never truncated. In the public suite that is 226 ToolRet requests and 1 BRIGHT request.

## Reproduce

The kit at `25c6ec7` and the model's code on the Python path:

```bash
hf download turion-ai/turion-jev-31b --local-dir /models/turion-jev-31b
pip install "/models/turion-jev-31b/code[vllm]" && pip install -e ".[transformers]"
python -m decision_index pipeline --engine turion_jev.kit_engine:TurionJev \
  --option model_dir=/models/turion-jev-31b --option backend=vllm --option quantization=fp8 \
  --option order_avg_max=2 --option second_pass_max_questions=1 --option repeat_max_questions=1 \
  --option warm_prefix=true --option recheck=false \
  --compact --out runs/turion-jev-31b
```

The model card has the details: [Run it for the Decision Index](https://huggingface.co/turion-ai/turion-jev-31b#run-it-for-the-decision-index).
