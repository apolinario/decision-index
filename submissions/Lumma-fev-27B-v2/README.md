Lumma-fev-27B-v2
Model: FrontiersMind/Lumma-fev-27B-v2 (full-weight fine-tune of Qwen3.8-27B; decision model, no text generation)
Edition: 0.3 (full public suite, 140,620 requests, 140,178 scoreable)
Decision Index: 63.4
scores.json: "complete": true
Results
Results dataset (public, pinned to tag v1.0): rohitsali/lumma-fev-27b-v2-decision-index
The run directory is runs/lumma-fev-27b-v2/ and contains results.jsonl.gz, scores.json, index.json, benchmark-summary.json, environment.json, status.json and source/.
In results.jsonl.gz the payload and raw_output fields are removed, because they contain benchmark text that some suite sources do not allow republishing. This copy re-scores to the same 63.4. The full file with those fields is available to the maintainers on request.
How it was run
Kit: stock http engine, decision-index commit f8552a1, no changes to the evaluation code or data.
Suite: built with suite rebuild at commit f8552a1 and suite import (pinned hashes verified), then restored from a private backup of that verified suite for this run.
Evaluated checkpoint: training run v3-bidir1, step 900. The training code commit is recorded in source/decision_config.json in the results dataset.
Model served with the autojev server.py (/v1/systemone endpoint). Package versions are in source/serve-env.txt.
Hardware: 4x NVIDIA H200 (vast.ai). 8 server processes (2 per GPU), 8 clients, one request at a time per client.
Data parallelism: the request set was split in 8 by sha256(run_id) mod 8, each shard run as a normal run --rows, results concatenated and scored once with score --edition 0.3.
Latency
Measured with one client, one request at a time, on one H200, over a stratified sample of 300 requests (290 ok rows used, the first 10 dropped as warm-up):

median
mean
p80
77.4 ms
554.9 ms
771.2 ms


These are not from the maintainers' sample on one RTX PRO 6000. The mean is raised by a few requests with very long inputs.
Declared limits
State + question must fit in 8,192 tokens. Longer inputs are refused by the server (HTTP 422, "no input was truncated"), not truncated.
369 of 140,620 requests ended as errors for this reason: ToolRet 335, POP909-CL 25, BRIGHT 6, API-Bank 2, HLE 1. They score zero where they fall in scored rows.
