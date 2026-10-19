# AutoTrust-JEV-27B v1.1

Self-run Decision Index 0.3 public score **67.32** (raw 75.62, breadth 66.47). Complete: 140,620 requests all ok; 140,178
scored after the official 442 exclusions. No errors, unsupported requests or abstentions. This is not the private Full
score or a maintainer-verified rank.

- Model: [autotrust/AutoTrust-JEV-27B-v1.1 @ `ae15690`](https://huggingface.co/autotrust/AutoTrust-JEV-27B-v1.1/tree/ae156900e9f338e423042b89e25d0fd8093064d4),
  a typed-decision model fine-tuned from [vllm-sr/d3](https://huggingface.co/vllm-sr/d3) (Apache-2.0; d3's notice kept as
  `NOTICE.base`), which is built on Qwen/Qwen3.8-27B @ `1d4bf0f`. bf16; the 255-row decision readout is folded into the
  `lm_head` rows of the 255 answer-code tokens (`vllm_readout.json`); full attention reads the prompt bidirectionally.
- Results: [runs/autotrust-jev-27b-v1.1](https://huggingface.co/datasets/autotrust/jev-decision-index-results/tree/7fe2f14c4bc30bdd77e99a92dd9b63f2981b607b/runs/autotrust-jev-27b-v1.1)
  (`results.jsonl.gz`, `scores.json`, `index.json`, `benchmark-summary.json`, `environment.json`, `latency.json`, the 16
  shard `status.json` files and the code that ran in `source/`). Re-scoring `results.jsonl.gz` with `score --edition 0.3`
  reproduces 67.32.
- Kit: apolinario/decision-index @ `f5a6e4fd85fed05e77878dfc4966d0568bfb45ec`, unchanged runner and scorer.
- Engine: `jev_vllm_engine:JevVllmEngine` with its prompt renderer `jev_format.py`, both shipped in the model repository
  (copies in this directory), against one vLLM server started with the repository's `serve_vllm.sh`.

| area (skill) | Knowledge & Reasoning | Language | Retrieval & Classification | Tools & Automation | Arts & Taste |
|---|---:|---:|---:|---:|---:|
| AutoTrust-JEV-27B v1.1 | 0.552 | 0.734 | 0.670 | 0.852 | 0.510 |

## Reproduce

Linux, one NVIDIA GPU with 80 GB or more (ran on one B200), Python 3.12, torch 2.13 + CUDA 13.2, vLLM 0.31.0,
transformers 5.17.0, and the open-source [StartLux-Decision](https://github.com/StartLuxLabs/StartLux-Decision) package
(Apache-2.0) at `cb1d11f1ac9e9b43b95ef3a6468fa29b607c40fd` for its vLLM prompt-bidirectional plugin.

```bash
pip install vllm==0.31.0 transformers==5.17.0 \
  "startlux-decide @ git+https://github.com/StartLuxLabs/StartLux-Decision@cb1d11f1ac9e9b43b95ef3a6468fa29b607c40fd"
pip install -e .                                   # this kit
hf download autotrust/AutoTrust-JEV-27B-v1.1 --revision ae156900e9f338e423042b89e25d0fd8093064d4 \
  --local-dir AutoTrust-JEV-27B-v1.1
bash AutoTrust-JEV-27B-v1.1/serve_vllm.sh AutoTrust-JEV-27B-v1.1   # vLLM on 127.0.0.1:8090
PYTHONPATH=AutoTrust-JEV-27B-v1.1 python -m decision_index run --engine jev_vllm_engine:JevVllmEngine \
  --option model=AutoTrust-JEV-27B-v1.1 --option url=http://127.0.0.1:8090/v1 --edition 0.3 --out runs/autotrust-jev-27b-v1.1
python -m decision_index score --edition 0.3 --results runs/autotrust-jev-27b-v1.1/results.jsonl
```

How this run was executed: the 0.3 run list (140,620 rows) was split into 16 parts by `sha256(run_id) mod 16`, and each
part ran as a normal kit run (`run --rows`, `run_shards.sh` in this directory), 16 processes against the one server. The
16 results files were concatenated (140,620 rows, 140,620 unique `run_id`s) and scored with `score --edition 0.3`.
`environment.json` is part 0's; the 16 parts differ only in `rows_path`. Wall time 2 h 30 min.

## Inference

- **Server.** `serve_vllm.sh` as published: `STARTLUX_PREFIX_BIDIR=1` (the plugin reads each prompt bidirectionally in
  the full-attention layers, as the model was trained; the Gated DeltaNet layers stay causal), `--dtype bfloat16
  --max-model-len 65536 --max-num-batched-tokens 65536 --no-enable-prefix-caching --no-enable-chunked-prefill
  --attention-backend FLASH_ATTN --logprobs-mode processed_logprobs --max-logprobs 255 --max-num-seqs 64
  --gpu-memory-utilization 0.85 --generation-config vllm`.
- **Read-out.** One completion per question with `max_tokens=1`, `allowed_token_ids` = the question's first n answer-code
  tokens and `logprobs=n`; the processed log-probabilities over that set, at temperature 1.0 (`vllm_readout.json`), are
  softmaxed into the option probabilities. Nothing is generated beyond that one token; no abstention outcome.
- **Questions.** Rendered by `jev_format.py`: the model's system prompt, the state, the question, every option under a
  single-token answer code. A request's questions are sent concurrently (8 per request).

## Latency

`latency.json`: one sequential client (the kit runner with this engine), the same server, one B200, 430 rows (10 per
dataset of the 0.3 run list): median **148 ms**, mean **360 ms**, p80 **348 ms** per request (p95 1,669 ms). The
`total_wall_ms` values in `results.jsonl.gz` were measured under 16 concurrent clients and include queueing (median 355 ms,
mean 1,007 ms); they are not single-request latencies.

## Capacity

Text only; question types `choice` and `noul` (the 0.3 public suite has 308,559 choice and 14,700 noul questions, at most
151 options). Up to 255 options per choice question in one pass, every option read and none pruned; context 65,536
tokens; nothing truncated and no request refused (0 unsupported, 0 errors).

## Suite

Rebuilt with `suite rebuild` and imported with `suite import`; `suite verify --edition 0.3`: selected rows, added rows,
GSM8K rows `75933489…`, exclusions `331df32d…` and subsets all matching.

## Training data

Fine-tuned from vllm-sr/d3 on TypeSafe Jev teacher distributions, ground-truth decision data built from public datasets
(including the public training splits of datasets whose test splits the suite uses) and rows of the public
[`tasksource/tasksource-jev-typed-decisions`](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions).
No test split of any suite benchmark was used. The training data of the d3 base is its authors'.

Hardware: 1 × NVIDIA B200 (183 GB). Not affiliated with TypeSafe AI.
