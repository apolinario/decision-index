# Wald-26B-A4B

[org2ai/Wald-26B-A4B](https://huggingface.co/org2ai/Wald-26B-A4B/tree/bc28ab1e074cdf963d1186e0f0956c96a7ee8b70) at `bc28ab1` is Gemma-4-26B-A4B-it (mixture of experts, 26B parameters, ~4B active) after full-parameter decision training (checkpoint `05901-c33`), Apache-2.0. The weights include the base model's unchanged vision tower.

- It reads the state and the questions through the Gemma 4 chat template and takes one probability per option from the option-letter logits, with no temperature table (T = 1).
- One pass, the setting of this run and the server default: no generated tokens. More than 26 options: a one-pass knockout over all options.

| | |
|---|---|
| Decision Index 0.3 public index | 62.30 (raw 71.50, breadth 61.03) |
| Requests | complete: 140,178 of 140,178 scoreable (140,620 sent), all answered, none unsupported, nothing truncated, no errors |
| Latency (serial) | median 53 ms per request: one request in flight, warm, processing time, CUDA graphs, one NVIDIA RTX PRO 6000, 200 Decision Index 0.3 requests |

## Run

```sh
hf download org2ai/Wald-26B-A4B --revision bc28ab1e074cdf963d1186e0f0956c96a7ee8b70 --local-dir ./Wald-26B-A4B && cd Wald-26B-A4B
./run.sh "$PWD"      # vLLM 0.30.0 + the reader in reference/ on :8000 (needs uv): one pass

git clone https://github.com/apolinario/decision-index && cd decision-index && git checkout 62d2f51de34a2de64906345b6bc3e98e27ff55c7
pip install -e .
decision-index pipeline --edition 0.3 --engine http --option base_url=http://127.0.0.1:8000 --out runs/Wald-26B-A4B
decision-index score --edition 0.3 --results results.jsonl.gz --out rescore      # the published file
```

- **Server:** `run.sh` (repository root) starts vLLM 0.30.0 (transformers 5.17.0, BF16, `VLLM_USE_FLASHINFER_SAMPLER=0`, `--max-model-len 131072 --gpu-memory-utilization 0.90 --max-num-seqs 128 --max-logprobs 64 --seed 0 --language-model-only`, CUDA graphs on) on loopback and the reader `python -m eval.systemone_vllm` (`reference/` in the model repo, `--template chat --prompt-format plain --wide knockout --gate 0`) on `POST /v1/systemone`. It checks every reader file against `reference/source-pins.json` and refuses to start if one differs. Exact settings: `serving.json` in the model repo. `EFFORT=auto` turns on an optional Auto 0.7 mode (native thought when unsure); it was not used for this run and is not part of this submission.
- **How the run was made:** a fresh 0.3 run with the kit at `62d2f51`, no rows carried over. The kit's `http` engine (`run --edition 0.3 --engine http --rows SHARD --compact`) sent every row of the 0.3 public suite to 4 reader front-ends, each over its own vLLM replica (one per GPU, the same flags and reader as `run.sh`, one pass), rows split into 256 shards by SHA256(run_id), 40 concurrent requests per replica. The shard result files were merged (one row per run_id) and scored with the unmodified `score --edition 0.3`. Re-scoring the published `results.jsonl.gz` with `62d2f51` gives 62.30.
- **Results:** [`org2ai/Wald-26B-A4B-decision-index-results`](https://huggingface.co/datasets/org2ai/Wald-26B-A4B-decision-index-results/tree/7629ef1d50aad82149d4df0786cc99d791da2dfb/runs/wald-26b-a4b-05901-c33-onepass) @ `7629ef1`; every answer and probability as returned (no field removed or shortened).
- **Weights:** the run read the training snapshot `05901-c33` (SHA256SUMS checked before the run); the released repository holds the same weight shards, language model and unchanged vision tower, with the base model's config and tokenizer files.
- **Declared limits:** 131,072-token context for state, question and options together; any number of options (more than 26: knockout); nothing truncated; none unsupported in this run.
- **Training data:** about 17 % of training tokens are the train splits of public benchmarks that the Decision Index also draws on, reformatted as decisions. The rest is broad decision data: task-specific decision sets, general decision tasks, decisions written and/or labelled by large frontier models (including robustness cases), web and GUI actions, and math. Public datasets are used as train splits only. No test row is included: the training data was scanned against the complete 0.3 public suite (0 strict hits).
- **Training:** one full-parameter decision-training pass from Gemma-4-26B-A4B-it; the submitted weights are the snapshot at two thirds of that pass, after 94.3M training tokens. Router, embeddings and vision tower frozen; no LoRA, no merge.
- **Vision:** not part of this submission. The weights contain the base model's vision tower unchanged, but the official path runs vLLM with `--language-model-only` and accepts text only; we have not evaluated image decisions.
