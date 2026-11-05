Lumma-fev-31B-v2
Model: FrontiersMind/Lumma-fev-31B-v2 (LoRA fine-tune of google/gemma-4-31B-it, rank 32, merged into the weights; decision model, no text generation)
Edition: 0.3 (full public suite, 140,620 requests, 140,178 scoreable)
Decision Index: 61.97
scores.json: "complete": true
Results
Results dataset (public, pinned to tag v1.0): rohitsali/lumma-fev-31b-v2-decision-index
The run directory is runs/lumma-fev-31b-v2/ and contains results.jsonl.gz, scores.json, index.json, benchmark-summary.json, environment.json, status.json and source/.
In results.jsonl.gz the payload and raw_output fields are removed, because they contain benchmark text that some suite sources do not allow republishing. This copy re-scores to the same 61.97. The full file with those fields is available to the maintainers on request.
How it was run
Kit: stock http engine, decision-index commit f8552a1, no changes to the evaluation code or data.
Suite: built with suite rebuild at commit f8552a1 and suite import (pinned hashes verified), then restored from a private backup of that verified suite for this run.
Model served with the AutoJev server.py (/v1/systemone endpoint) from the lora-gemma4 branch of FrontiersMindAI/autojev. The exact commit and the model revision are in source/RUN_NOTES.md, and package versions in source/serve-env.txt.
Hardware: 4x NVIDIA H100 NVL 94 GB (vast.ai). 4 server processes (1 per GPU), 4 clients, one request at a time per client.
Data parallelism: the request set was split in 4 by sha256(run_id) mod 4, each shard run as a normal run --rows, results concatenated and scored once with score --edition 0.3.
Latency
Measured with one client, one request at a time, on one H100 NVL, over a stratified sample of 300 requests (290 ok rows used, the first 10 dropped as warm-up):

median
mean
p80
132.8 ms
1095.7 ms
1588.0 ms


On this hardware the mean and the p80 are above 1000 ms. These figures are not from the maintainers' sample on one RTX PRO 6000.
Declared limits
State + question must fit in 8,192 tokens. Longer inputs are refused by the server (HTTP 422), not truncated.
371 of 140,620 requests ended as errors for this reason: ToolRet 315, POP909-CL 48, BRIGHT 6, API-Bank 2. They score zero where they fall in scored rows.
