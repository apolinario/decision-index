# Decision Index submissions

# Seb-9B-v2

- Model: [ironbcc/Seb-9B-v2](https://huggingface.co/ironbcc/Seb-9B-v2), Apache-2.0, Qwen3.5-9B fine-tune.
- Results: [public 0.3 run](https://huggingface.co/datasets/ironbcc/seb-9b-v2-decision-index/tree/main/runs/seb-9b-v2-0.3), including `scores.json`, compact `results.jsonl.gz`, environment/provenance and inference source.
- Kit: [`9eb2dbe2a358004c8782c66e40a83ac07b953fec`](https://github.com/apolinario/decision-index/commit/9eb2dbe2a358004c8782c66e40a83ac07b953fec), `http` engine, edition 0.3.
- Hardware: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB per GPU. Original indexed benchmarks used two independent single-GPU PyTorch BF16 shards; completion of missing non-index benchmarks was started with vLLM BF16 on one GPU and checkpointed for this draft. Original answers are preserved; runtimes and scheduling are disclosed in the results dataset.
- Inference: fixed candidate-v2 prompt, thinking disabled, full candidate probabilities normalized from next-token scores, no truncation or option filtering. Context 32,768 tokens; payload cap 262,144 characters. vLLM 0.30.0: processed_logprobs, max_logprobs=20, generation_config=vllm, max_num_seqs=64, gpu_memory_utilization=0.45; readout max_tokens=1, temperature=1, top_p=1, top_k=-1, allowed_token_ids restricted to all supplied option keys.
- Reproduction: `REPRODUCTION.md` and `inference/` in the results directory document exact commands, pinned model revision, weight/source hashes, ordinary single-stream serving and the bounded concurrency wrapper used for completion.
- Public index: 59.16 (self-scored). No private/Full score claimed; requesting maintainer evaluation.

**Draft / not eligible yet:** the full public run is incomplete (`scores.json` has `complete: false`). Please do not add a final leaderboard entry yet. The missing non-index benchmark requests must be finished and the run re-scored before review.

Checkpoint: 112,059/140,178 scoreable requests complete; 28,119 pending. Completion is stopped and resumable.
