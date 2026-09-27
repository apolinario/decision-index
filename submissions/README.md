# Submissions

| Model | Decision Index | Results | Engine and exact code | Hardware | Declared capacity limits |
|---|---:|---|---|---|---|
| Lavoir (`moganai/lavoir` @ `4c5eaeb`) | **8.16** | [full run and scores](https://huggingface.co/datasets/moganai/lavoir-decision-index-results/blob/2a9f764674436a1957f517312711bbb5878daff4/runs/lavoir/scores.json) | `di_lavoir_engine:LavoirEngine` with `native=true` (in the results dataset under `runs/lavoir/harness/`); kit `19ad28e` (0.2); [lavoir](https://github.com/moganai/lavoir) `af79b0a`, laya 0.3.11 | 1 JUPITER node, 4 x NVIDIA GH200; 4 runner processes (one per GPU) | Context 8,192 tokens; options read with Laya's native 48-token per-option budget, as Laya was run. 331 requests that do not fit 8,192 tokens are unsupported. |
