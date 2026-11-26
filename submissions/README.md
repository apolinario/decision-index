# Model submissions

| Track | Model | Model repository and revision | Public results and revision | Inference code and commit | Author Public Vision Index | Engine / kit | Hardware and settings |
|---|---|---|---|---|---:|---|---|
| vision | Jev-CLP-0.8B-v1.0 | [model](https://huggingface.co/Nicholas-WKK/Jev-CLP-0.8B-v1.0/tree/a5df30c9505858d338e98534fd0545c51aaa21cf) | [vision-scores.json](https://huggingface.co/datasets/Nicholas-WKK/Jev-CLP-0.8B-v1.0-results/blob/85c600affb863c397dafa33c78da466e131e5644/runs/Jev-CLP-0.8B-v1.0/vision-scores.json) | [code](https://github.com/Nicholas-WKK/jev-clp/tree/e641e215e07d1f326bb90c895846436962914e2b) | 61.439927156678166 | `jev_clp.engine:VisionEngine`; kit `25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d` | 8 × A100-SXM4-40GB; batch 4/GPU; BF16/SDPA; min_pixels 784, max_pixels 1638400; no truncation. All three repositories private; maintainer access pending coordination. See PR for exact reproduction and training-data disclosure. |
