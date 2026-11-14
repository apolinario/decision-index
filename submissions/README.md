# Submissions

| Board | Model | Hub repo | Results | Engine and options | Kit commit | Hardware | Resolution | Declared limits |
|---|---|---|---|---|---|---|---|---|
| **vision** | 𒉾𒄑𒄀 (peshgi) | [Model](https://huggingface.co/plantwaterer/peshgi/tree/6e8915481b58016d3b0186eda634801d85846a72) | [Run](https://huggingface.co/plantwaterer/peshgi/tree/6e8915481b58016d3b0186eda634801d85846a72/runs/peshgi-vision-0.3) ([notes](peshgi/README.md)) | `peshgi:Engine`; no options | `25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d` | 1 x RTX PRO 6000 96GB (ten benchmarks), 1 x H200 (Winoground); BF16 | Native up to 1,638,400 pixels per image; no lower cap set | Choice only; one question per request; 2–12 options; no truncation; no image-count cap |
