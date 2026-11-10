# Submissions

| Board | Model | Hub repo | Results | Engine and options | Kit commit | Hardware | Resolution | Declared limits |
|---|---|---|---|---|---|---|---|---|
| **vision** | dormice | [Model](https://huggingface.co/plantwaterer/dormice/tree/13caf7d0503a14fe6e0b8d3570a76ea88acdcf48) | [Run](https://huggingface.co/plantwaterer/dormice/tree/13caf7d0503a14fe6e0b8d3570a76ea88acdcf48/runs/dormice-vision-0.3) ([notes](dormice/README.md)) | `dormice:Engine`; no options | `25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d` | 1 x RTX PRO 6000 96GB (ten benchmarks), 1 x H200 (Winoground); BF16 | Native up to 1,638,400 pixels per image; no lower cap set | Choice only; one question per request; 2–12 options; no truncation; no image-count cap |
