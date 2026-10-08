# Submissions

| model | model repo | results | engine / code | hardware | settings |
|---|---|---|---|---|---|
| Darwin-27B-ZTC | [FINAL-Bench/Darwin-27B-ZTC](https://huggingface.co/FINAL-Bench/Darwin-27B-ZTC) @ f0ffe63 (weights) | [FINAL-Bench/Darwin-27B-ZTC-decision-index/runs/darwin-27b-ztc](https://huggingface.co/datasets/FINAL-Bench/Darwin-27B-ZTC-decision-index/tree/main/runs/darwin-27b-ztc) (public 0.3: 62.63, complete) | kit 0.3 `http` engine (9eb2dbe) -> `ztc_server.py` in the model repo; in-process alternative `ztc_engine:DarwinZTCEngine` | 2x B200 (4 server processes) | `ZTC_MODEL=FINAL-Bench/Darwin-27B-ZTC python ztc_server.py`; one forward pass per question, no thinking, no generated tokens; batch 8, token budget 16000, max length 32768; longer inputs refused (none in the suite), nothing truncated; 0 errors, 0 unsupported |
