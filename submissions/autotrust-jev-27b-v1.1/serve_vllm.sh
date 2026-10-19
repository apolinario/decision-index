#!/usr/bin/env bash
# usage: serve_vllm.sh [MODEL_DIR]; needs vllm==0.31.0 and startlux-decide (see README.md)
set -euo pipefail
MODEL_DIR=${1:-$(dirname "$0")}
MAX_LEN=${MAX_LEN:-65536}
export STARTLUX_PREFIX_BIDIR=1
# FlashInfer's sampler compiles with nvcc at first use; images without nvcc crash on it
export VLLM_USE_FLASHINFER_SAMPLER=0
exec vllm serve "$MODEL_DIR" --host "${HOST:-127.0.0.1}" --port "${PORT:-8090}" --dtype bfloat16 \
  --max-model-len "$MAX_LEN" --max-num-batched-tokens "$MAX_LEN" --no-enable-prefix-caching --no-enable-chunked-prefill \
  --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":0,"video":0}' --logprobs-mode processed_logprobs \
  --max-logprobs 255 --attention-backend FLASH_ATTN --served-model-name AutoTrust-JEV-27B-v1.1 --max-num-seqs 64 \
  --generation-config vllm
