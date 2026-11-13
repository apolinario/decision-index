# Submissions

| Board | Model | Hub repo | Results | Engine and options | Kit commit | Hardware | Resolution | Declared limits |
|---|---|---|---|---|---|---|---|---|
| **vision** | CoCo-Decision-Vision-0.8B | [Model](https://huggingface.co/corners-ai/CoCo-Decision-Vision-0.8B/tree/fba7ec35d1ef48c925f13e0173122a045f3cac68) | [Run](https://huggingface.co/corners-ai/CoCo-Decision-Vision-0.8B/tree/fba7ec35d1ef48c925f13e0173122a045f3cac68/runs/CoCo-Decision-Vision-0.8B-vision-0.3) ([notes](CoCo-Decision-Vision-0.8B/README.md)) | `decide`; `protocol=decisions`, `path=/v1/systemone`, `base_url` of the package's `d3_server.py`; `--workers 4` | `25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d` | 1 x NVIDIA GB10 (DGX Spark); BF16 backbone, FP32 readout | Native up to 1,638,400 pixels per image (kit cap; the package's own cap is also 1,638,400) | None beyond d3-lite's: no image-count cap; no truncation |
