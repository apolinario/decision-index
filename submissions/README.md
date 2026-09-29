# Submissions

| Model | Decision Index | Results | Engine and exact code | Hardware | Declared capacity limits |
|---|---:|---|---|---|---|
| Von 1.3 (`wfzyx/von`, ModernBERT-large option-marker encoder, 395M; PyPI `von-sdk==1.3.2`, engine commit `a867a49`, weights `wfzyx/von` @ `5df8185`) | **13.74** | [full run and scores](https://huggingface.co/datasets/wfzyx/decision-index-results/blob/edebb3a6f0bdc67eb99baf4a737ca02a8c71e3d5/runs/von-1.3/scores.json) | kit `http` engine against `von serve --device openvino:cuda --on-overflow refuse` (native `/v1/systemone`, one request per row, chain-of-options on by default); kit `87d4650` (0.2.1 scoring) | 1 x NVIDIA A10G (AWS g5.xlarge), OpenVINO CUDA fp16 | Context 8,192 tokens, nothing truncated: 323 requests that do not fit are refused with HTTP 422 and recorded unsupported (7 scoreable). |
