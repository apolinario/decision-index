# Submissions

| Board | Model | Hub repo | Results | Engine and options | Kit commit | Hardware | Resolution | Declared limits |
|---|---|---|---|---|---|---|---|---|
| **vision** | autorsi-jev-4b | [Model](https://huggingface.co/k2021211255/autorsi-jev-4b/tree/c135afb09cb279b3045bca130ddaae1814167a91); revision `v1` | [Complete run](https://huggingface.co/datasets/k2021211255/autorsi-jev-4b-Results/tree/3bba8fb24c449eb65646ee66e1c32e4da6b31582/runs/autorsi-jev-4b) | `decide`; `base_url=http://127.0.0.1:8000`; `protocol=decisions`; `path=/v1/decisions`; `model=autorsi-jev-4b`; `image_encoding=data_url`; `max_pixels=1638400`; `workers=1` | `25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d` | Single A800 80GB; BF16; sequential | Native up to 1,638,400 pixels per image; lower caps lifted | 32,768 tokens; up to 255 options; no image-count cap; explicit capacity errors |
