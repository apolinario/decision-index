# Submissions

| Board | Model | Hub repo | Results | Engine and options | Kit commit | Hardware | Resolution | Declared limits |
|---|---|---|---|---|---|---|---|---|
| **vision** | glires | [Model](https://huggingface.co/plantwaterer/glires/tree/adc4e87067541aa6289147888f5fee2dde11f077) | [Run](https://huggingface.co/plantwaterer/glires/tree/adc4e87067541aa6289147888f5fee2dde11f077/runs/glires-vision-0.3) ([notes](glires/README.md)) | `glires:Engine`; no options | `25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d` | 1 x RTX PRO 6000 96GB (ten benchmarks), 1 x H200 (Winoground); BF16 | Native up to 1,638,400 pixels per image; no lower cap set | Choice only; one question per request; 2–12 options; no truncation; no image-count cap |
