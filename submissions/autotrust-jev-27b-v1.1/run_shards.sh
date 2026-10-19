#!/bin/bash
# AutoTrust-JEV-27B-v1.1, Decision Index 0.3 public suite: 16 kit runs (one per shard of the run list, split by sha256 of run_id)
# against one vLLM server (serve_vllm.sh), with the model's own engine jev_vllm_engine:JevVllmEngine; kit f5a6e4f.
cd /root/di-runs-fast/autotrust-jev-27b-v1.1
export PYTHONPATH=/root/models/AutoTrust-JEV-27B-v1.1
for i in $(seq -w 0 15); do
  /root/jev11/bin/python -m decision_index run --engine jev_vllm_engine:JevVllmEngine \
    --option model=/root/models/AutoTrust-JEV-27B-v1.1 --option url=http://127.0.0.1:8090/v1 \
    --edition 0.3 --rows shards/rows_$i.jsonl --out parts/p$i > logs_p$i.log 2>&1 &
done
wait
echo ALL_SHARDS_DONE
