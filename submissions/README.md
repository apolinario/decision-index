# Submissions

| Model | Decision Index | Results | Engine and exact code | Hardware | Declared capacity limits |
|---|---:|---|---|---|---|
| Eikos-27B-FP8 (`caiovicentino1/Eikos-27B-FP8` @ `e40a92f`) | **55.46** (0.2.1); 51.52 (0.2) | [full run and scores](https://huggingface.co/datasets/caiovicentino1/Eikos-27B-FP8-decision-index/blob/c0ae08566bc18d54548bd41257cb35ce1d3cae27/runs/eikos-27b-fp8/0.2.1/scores.json) | `http` engine against the model repo's own `serve.py` (v1.2) on vLLM 0.30.0; launch and run scripts in the results dataset under `runs/eikos-27b-fp8/harness/`; run with kit `19ad28e` (0.2), rescored with `87d4650` (0.2.1) | 1 x NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB); one runner, one request at a time | Context 65,536 tokens (the longest prompt is 31,696); no request unsupported. |
