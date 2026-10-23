# Submissions

One line per submitted model. Runs must be complete (`scores.json` says `"complete": true`) and untouched.

| Model | Model repo | Results dataset | Engine / commit | Hardware | Exact inference settings |
|---|---|---|---|---|---|
| Koa-Decisions-v0.1 (text, 31B; R7a-T2) | [SF-Foundation-DS/Koa-Decisions-v0.1](https://huggingface.co/SF-Foundation-DS/Koa-Decisions-v0.1) (private; see access note below) | [SF-Foundation-DS/Koa-Decisions-v0.1-results](https://huggingface.co/datasets/SF-Foundation-DS/Koa-Decisions-v0.1-results) (`runs/koa-decisions-v0pt1-vllm032/scores.json`, private) | vLLM 0.32.0 stable (torch 2.13.0+cu130, transformers 5.19.0) with the prefix-bidirectional engine in the model repo's `submission/` folder (`di_vllm_bidir.py`, a kit `Engine`); public suite scored with this kit at `25c6ec7` | 6x NVIDIA H200 (public run; shards in parallel) | bf16; prefix cache on with a per-prompt `cache_salt` (sha1 of the prompt ids); chunked prefill off; TRITON_ATTN; Model Runner V1 (`VLLM_USE_V2_MODEL_RUNNER=0`); Triton `mm_prefix` ranges on the full-attention layers (sliding layers causal); `max_model_len` 32768; PIECEWISE CUDA graphs; option score = key tokens + `<eos>` summed log-prob, softmax, argmax (the reported probabilities use a fitted calibration temperature T = 1.15, `DI_TEMPERATURE=1.15`; it does not change the argmax or any score); weights used as released, no pull-back; over-long prompts are refused, never truncated |

Access note for Koa-Decisions-v0.1: both Hub repos are private, in the `SF-Foundation-DS` organisation. `multimodalart` is being added as a read-only member. Please comment on the pull request if anyone else needs access.
