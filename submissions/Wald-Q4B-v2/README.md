# Wald-Q4B v2.1

[org2ai/Wald-4B](https://huggingface.co/org2ai/Wald-4B/tree/5fbae66abfca1fe8c60c5a0877d12593e6dd1854) at tag `v2.1` = `5fbae66` is Qwen3.5-4B (chat) after one broad full-parameter decision-training pass from the chat model, lightly merged back toward the chat weights (checkpoint `056A0`), Apache-2.0, plus the base model's unchanged vision tower. It replaces v2.0 (`919a9d7`, 54.18).

- It reads the state and the questions through the Qwen chat template and takes one probability per option from the option-letter logits, calibrated with a frozen temperature table (`temperature.json`, sha256 `a0f72cd2…`).
- Auto 0.7, the setting of this run: if a question's top probability is below 0.7, the model thinks once natively (at most 512 tokens) and the options are read again. More than 26 options: a one-pass knockout over all options, no thought.

| | |
|---|---|
| Decision Index 0.3 public index | 60.05 (raw 70.12, breadth 58.63) |
| Requests | complete: 140,178 of 140,178 scoreable (140,620 sent), all answered, none unsupported, nothing truncated, no errors |
| Latency (serial) | median 158 ms per request: one request in flight, warm, processing time, one NVIDIA RTX PRO 6000, 200 Decision Index 0.3 requests |

## Run

```sh
hf download org2ai/Wald-4B --revision 5fbae66abfca1fe8c60c5a0877d12593e6dd1854 --local-dir ./Wald-Q4B-v2.1 && cd Wald-Q4B-v2.1
./run.sh "$PWD"      # vLLM 0.30.0 + wald-serve-native-vision on :8000 (needs uv): Auto 0.7, repeat_state_plain, budget 512

git clone https://github.com/apolinario/decision-index && cd decision-index && git checkout 62d2f51de34a2de64906345b6bc3e98e27ff55c7
pip install -e .
decision-index pipeline --edition 0.3 --engine http --option base_url=http://127.0.0.1:8000 --out runs/Wald-Q4B-v2.1
decision-index score --edition 0.3 --results results.jsonl.gz --out rescore      # the published file
```

- **Server:** `run.sh` starts vLLM 0.30.0 (transformers 5.17.0, BF16, `VLLM_USE_FLASHINFER_SAMPLER=0`, `--max-model-len 131072 --gpu-memory-utilization 0.85 --max-num-seqs 128 --seed 0`) and `wald-serve-native-vision` (`server/` in the model repo) on `POST /v1/systemone`. It pins the reader files and the calibration table and refuses to start if either differs. Exact settings: `RUNBOOK.md` in the model repo.
- **How the run was made:** a fresh 0.3 run with the kit at `62d2f51`, no rows carried over. It used the kit's Engine API with the adapter `evaluation/v2/code/di03_native_auto_engine.py` from the model repo (same reader, calibration and settings as the server) against six vLLM replicas, one per GPU, rows split by SHA256(run_id) across the replicas, 128 concurrent requests each. The result files were concatenated replica by replica and scored with the unmodified `score --edition 0.3`. Re-scoring the published `results.jsonl.gz` with `62d2f51` gives 60.05.
- **Results:** [`org2ai/Wald-Q4B-decision-index-results`](https://huggingface.co/datasets/org2ai/Wald-Q4B-decision-index-results/tree/8deab0b1cf157b8a590ce7298e21b2dd44ee04dc/runs/wald-q4b-v21-056A0-auto07) @ `8deab0b`; only the generated thought text was removed from `results.jsonl.gz`, every answer and probability is unchanged.
- **Weights:** the run used the text-only weights of `056A0`; the released repository holds the same language-model tensors byte for byte plus the base model's vision tower (release vs text-only on 3,125 held-out decision items: same answer on 3,110, max probability difference 0.031).
- **Declared limits:** 131,072-token context for state, question and options together; any number of options (more than 26: knockout); nothing truncated; none unsupported in this run.
- **Training data:** about 23 % of training tokens are the train splits of public benchmarks that the Decision Index also draws on, reformatted as decisions. The rest is broad decision data: general decision tasks, web and GUI actions, task-specific decision sets, and decisions written and/or labelled by large frontier models, including robustness cases. Public datasets are used as train splits only. No test row is included: every training file was gated against the complete 0.3 public suite before training (0 hits). The checkpoint was chosen on 20 held-out training-source families that were never trained on, not on the public index.
- **Vision:** the official path takes images (state message parts or a top-level `images` list of data URIs, up to 16 per request) through the unchanged vision tower, zero-shot.
