#!/bin/bash
# The script exactly as run (paths on our machine). Generic command: see README.md, "Reproduce".
# AutoTrust-JEV-27B-v1.1 on the Decision Index 0.3 public vision suite (kit f5a6e4f), in-process vLLM 0.31.0 + StartLux plugin.
cd /root/di-vision-official
export PYTHONPATH=/root/di-vision-official
/root/jev11/bin/python -m decision_index vision run --dir work/vision --engine jev_vllm_vision_engine:JevVllmVision \
  --option model=/root/models/AutoTrust-JEV-27B-v1.1 --chunk 256 "$@" --out runs/autotrust-jev-27b-v1.1-vision
