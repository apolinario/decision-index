# Wald-Q4B v2

[org2ai/Wald-4B](https://huggingface.co/org2ai/Wald-4B/tree/919a9d72a65379271005ed0945bfc72742dad3af) at tag `v2.0` = `919a9d7` is Qwen3.5-4B (chat) after full-parameter decision training (checkpoint `04701-c22`), Apache-2.0, plus the base model's unchanged vision tower.

- It reads the state and the questions through the Qwen chat template and takes one probability per option from the option-letter logits, calibrated with a frozen temperature table (`temperature.json`, sha256 `a0f72cd2…`).
- Auto 0.7, the setting of this run: if a question's top probability is below 0.7, the model thinks once natively (at most 512 tokens) and the options are read again. More than 26 options: a one-pass knockout over all options, no thought.

| | |
|---|---|
| Decision Index 0.3 public index | 54.18 (raw 65.65, breadth 52.45) |
| Requests | complete: 140,178 of 140,178 scoreable (140,620 sent), all answered, none unsupported, nothing truncated, no errors |
| Latency (serial) | median 148 ms, mean 478 ms per request: one request in flight, warm, processing time, one NVIDIA RTX PRO 6000, 750 Decision Index 0.3 requests, the packaged server |
| Request time in `scores.json` | median 1,880 ms: 128 concurrent requests per replica, so it includes queueing; not a serial measurement |

## Run

```sh
hf download org2ai/Wald-4B --revision 919a9d72a65379271005ed0945bfc72742dad3af --local-dir ./Wald-Q4B-v2 && cd Wald-Q4B-v2
./run.sh "$PWD"      # vLLM 0.30.0 + wald-serve-native-vision on :8000 (needs uv): Auto 0.7, repeat_state_plain, budget 512
# GET /health: "engine": "native-v2", "checkpoint": "04701-c22", "policy": "Auto0.7", "temperature_sha256": "a0f72cd2…"

git clone https://github.com/apolinario/decision-index && cd decision-index && git checkout 62d2f51de34a2de64906345b6bc3e98e27ff55c7
pip install -e .
decision-index pipeline --edition 0.3 --engine http --option base_url=http://127.0.0.1:8000 --out runs/Wald-Q4B-v2
decision-index score --edition 0.3 --results results.jsonl.gz --out rescore      # the published file
```

- **Server:** `run.sh` starts vLLM 0.30.0 (transformers 5.17.0, BF16, `VLLM_USE_FLASHINFER_SAMPLER=0`, `--max-model-len 131072 --gpu-memory-utilization 0.85 --max-num-seqs 128 --seed 0`) and `wald-serve-native-vision` (`server/` in the model repo) on `POST /v1/systemone`. It pins the reader files and the calibration table and refuses to start if either differs. Exact settings: `RUNBOOK.md` in the model repo.
- **How the run was made:** a fresh 0.3 run with the kit at `62d2f51`, no rows carried over. It used the kit's Engine API with the adapter `evaluation/v2/code/di03_native_auto_engine.py` from the model repo (same reader, calibration and settings as the server) against four vLLM replicas, one per GPU, rows split by SHA256(run_id) mod 4, 128 concurrent requests each (`evaluation/v2/code/di03_parallel.py`). The four result files were concatenated replica by replica and scored with the unmodified `score --edition 0.3`. Re-scoring the published `results.jsonl.gz` with `62d2f51` gives 54.18.
- **Server vs adapter:** the packaged server on the released weights reproduces the run's answers on 682 of 692 questions of a 300-request sample (the differences come from sampled thoughts and gate decisions next to 0.7); one pass reproduces them on 230 of 231 JevBench public items.
- **Weights:** the run used the text-only weights file of `04701-c22`; the released repository holds the same language-model tensors byte for byte plus the base model's vision tower.
- **Declared limits:** 131,072-token context for state, question and options together; any number of options (more than 26: knockout); nothing truncated; none unsupported in this run.
- **Training data:** train splits of public decision, NLI, QA and tool-use datasets, including the train splits of Decision Index benchmarks in their request formats, our programmatic generators, synthetic data generated and/or labelled by large frontier models, and the model's own earlier answers. No test row is included: the training data was gated against the public suite before training, and a strict scan of every training file against the complete 0.3 public suite found 0 hits (`contamination/trained-on-di-ids.json` in the model repo).
- **Vision:** the official path takes images (state message parts or a top-level `images` list of data URIs, up to 16 per request) through the unchanged vision tower, zero-shot.
