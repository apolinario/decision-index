# Submissions

| Board | Model | Hub repo | Results | Engine and options | Kit commit | Hardware | Resolution | Declared limits |
|---|---|---|---|---|---|---|---|---|
| **vision** | miniRSI-jev-4b | [Model](https://huggingface.co/k2021211255/miniRSI-jev-4b/tree/762d06239315e98744fc61f65a0e570ad83da6c8); revision `v1.1` | [Complete run](https://huggingface.co/datasets/k2021211255/miniRSI-jev-4b-Results/tree/7754eaae4811bfe042739846fa79c654f3e658a7/runs/miniRSI-jev-4b) | `decide`; `base_url=http://127.0.0.1:8000`; `protocol=decisions`; `path=/v1/decisions`; `model=miniRSI-jev-4b`; `image_encoding=data_url`; `max_pixels=1638400`; `workers=1` | `25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d` | Single A800 80GB; BF16 | Native up to 1,638,400 pixels per image | Choice only; 2–26 options; 262,144 tokens; no image-count cap; explicit capacity errors |
