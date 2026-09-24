# Data formats

## Suite rows (`selected-rows.jsonl.gz`, `added-rows.jsonl.gz`)

One JSON object per request. Edition 0.2 has two row files, the base rows and the seven new benchmarks, and the runner reads both. Keys common to every row:

| Key | Meaning |
|---|---|
| `id`, `family`, `split` | source case id, benchmark family (track file stem) and upstream split |
| `state` | what the engine sees as context: a string or a JSON object; may be empty |
| `questions` | map of question key to `{"type": "choice", "instructions": str or object, "criteria": {option_key: description}}` (2 to 255 options), or `{"type": "noul", ...}` for a yes/no probability (RAGTruth, and the PhishNChips boolean questions; answered as `{"type": "noul", "noul": p_yes}`). Every question in the base rows is a `choice` |
| `expected` | gold answer per question key (`null` for unjudged retrieval candidates and for the PhishNChips signal questions, which are not scored) |
| `metadata` / `provenance` | source hashes, group ids, subject/song/user ids used by macro metrics; never sent to an engine |
| `scoring` | present on benchmarks with a custom scorer: `retrieval_ranking` (qrels, `field_to_document`), `routing_outcome` (outcomes per option), `chess_reference_value` (values, accepted ties), `human_group_rank` (summed ranks, accepted), `forecast_probability`; the new 0.2 benchmarks carry `candidate_decisions` or `calibration_decisions` with `primary_field` and `unscored_fields` |
| `_evaluation` | `run_id`, `catalog_id`, `dataset`, `group_id` (linked-case id), `track` (file stem), `source_path`, `payload_sha256` (hash of `{"model": "jev-1.13.0", "state", "questions"}`), `proxy_tokens`, `benchmark_origin` |

An engine receives exactly `state` and `questions`.

## Result rows (`results.jsonl`)

The runner writes one line per request in the lab's format:

```json
{"run_id": ..., "catalog_id": 24, "dataset": "MMLU", "group_id": ..., "track": ..., "source_path": ..., "payload_sha256": ..., "proxy_tokens": ..., "benchmark_origin": ...,
 "payload": {"state": ..., "questions": ...},
 "started_utc": "...", "engine": "transformers",
 "status": "ok" | "unsupported" | "error" | "abstained",
 "response": {"model": ..., "answers": {key: {"type": "choice", "choice": key, "probabilities": {key: p}}}, "usage": {"input_tokens": n}},
 "raw_output": <engine-native output>,
 "error": "...", "exception": "ClassName", "traceback": "...",
 "completed_utc": "...", "total_wall_ms": 106.1, "model_request_wall_ms": 106.1}
```

`response`/`raw_output` exist only for `ok`; `error` for the other statuses; `exception`/`traceback` for `error`. `--compact` drops `payload` and `raw_output`. `total_wall_ms` and `model_request_wall_ms` are the device-synchronized wall time of the engine call including prompt construction (identical unless an engine reports a separate model time); `http_wall_ms` is added by engines that measure HTTP separately.

Alongside: `status.json` (last event: loading, ready, progress, complete, failed), `environment.json` (engine options, provenance, torch/cuda/gpu, corpus hash, latency definition).

## Scores

- `benchmark-summary.json`: per benchmark `requests`, `answered`, `unsupported`, `errors`, `abstained`, `pending`, `scored_requests` (complete supported case groups), `metric`, `score`, `median_ms`, optional `tracks` and `detail`.
- `index.json` (0.2): `index` (the Decision Index, chance-corrected), `raw_index`, `scores` (`balanced_skill`, `balanced_raw`, `breadth_skill`), `areas` (five areas: raw, skill, coverage, n, benchmark ids), `benchmarks` (per benchmark: raw, skill, coverage, chance as `random`, `rule` = `track`, `chance` or `vs baseline`, `in_index`, tracks where there are several), `note`.
- `index.json` (0.1): `index` (balanced_raw), `scores` (three formulas), `areas`, `benchmarks`, `frozen_panel` (25-benchmark lower bound), `formulas`.
- `scores.json`: everything above plus edition, suite hashes, counts, latency and the completion flag; this is the file a submission points at. A run is complete when every scoreable request of the edition (0.2: 120,615 + 30,419) has a result.
