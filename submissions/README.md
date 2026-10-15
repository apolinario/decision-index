# Submissions

| Model | Decision Index | Results | Engine and exact code | Hardware | Declared capacity limits |
|---|---:|---|---|---|---|
| mogan-decision-31B-it (`moganai/mogan-decision-31B-it` @ `676d52d`, LoRA on `google/gemma-4-31B-it` @ `842da37`) | **65.42** (0.2.1 public index); 60.17 (0.2) | [full run and scores](https://huggingface.co/datasets/moganai/mogan-decision-31B-it-decision-index-results/blob/e751b050c5fbafd07f9b1a01d59f3debaf2df75e/runs/mogan-decision-31B-it/edition-0.2.1/scores.json) | `di_gemma_engine:GemmaEngine` (in the model repo under `code/`); kit `19ad28e` runner, rescored with `87d4650` (0.2.1) | JUPITER, 32 x NVIDIA GH200 (32 shards, one request at a time per GPU); latency per request: median 107 ms, mean 363 ms, 80th percentile 219 ms | none reached: 2–255 options, prompts up to 32,768 tokens; all 151,476 requests `ok` |
