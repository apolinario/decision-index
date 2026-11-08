# Submissions

| Board | Model | Hub repo | Results | Engine and options | Kit commit | Hardware | Resolution | Declared limits |
|---|---|---|---|---|---|---|---|---|
| Vision | LingeeJevMulti-4B-v1 | [Ashan0525/LingeeJevMulti-4B-v1](https://huggingface.co/Ashan0525/LingeeJevMulti-4B-v1) @ 02fb8fc | [Ashan0525/LingeeJevMulti-DecisionIndex-Vision-Results](https://huggingface.co/datasets/Ashan0525/LingeeJevMulti-DecisionIndex-Vision-Results) `runs/LingeeJevMulti-4B-v1-public/` @ 5792811 (public vision index 72.60) | `decide`, `protocol=decisions`, `path=/v1/decisions`, `image_encoding=data_url`, `max_pixels=1638400`; server `python systemone.py --model <dir> --port 18771` | 25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d | 1x NVIDIA H200 141GB, bf16, single process | native, capped at 1,638,400 px | none |
