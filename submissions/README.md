# Submissions

| Board | Model | Hub repo | Results | Engine and options | Kit commit | Hardware | Resolution | Declared limits |
|---|---|---|---|---|---|---|---|---|
| **vision** | Aster-JEV-27B | [Model](https://huggingface.co/k2021211255/Aster-JEV-27B/tree/6773d6c04e0278aa930221f5afcf60fad3fb1c21); server `decide-http-v1` | [Complete run](https://huggingface.co/datasets/k2021211255/Aster-JEV-27B-Results/tree/0d6a64c51c8d7de3898c41f6b67d1441931a0111/runs/Aster-JEV-27B-http-full) | `decide`; `base_url=http://127.0.0.1:8000`; `protocol=decisions`; `path=/v1/decisions`; `model=Aster-JEV-27B`; `image_encoding=data_url`; `max_pixels=1638400`; `workers=1` | `25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d` | Single A800 80GB; BF16 | Native up to 1,638,400 pixels per image; lower author-set caps lifted | Choice only; 2–26 options; 262,144 tokens; no image-count cap; explicit capacity errors |
