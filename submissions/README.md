# Submissions

| Board | Model | Hub repo | Results | Engine and options | Kit commit | Hardware | Resolution | Declared limits |
|---|---|---|---|---|---|---|---|---|
| **vision** | peshtur | [Model](https://huggingface.co/plantwaterer/peshtur/tree/73e6683622487fbc0aa8046afeb553b7d805bcaf) | [Run](https://huggingface.co/plantwaterer/peshtur/tree/73e6683622487fbc0aa8046afeb553b7d805bcaf/runs/peshtur-vision-0.3) ([notes](peshtur/README.md)) | `peshtur:Engine`; no options | `25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d` | H200, one GPU per process (scored run); 1 x RTX PRO 6000 96GB (latency); BF16 | Every image resized to 589,824–1,638,400 pixels (smaller images enlarged) | Choice only; one question per request; 2–26 options; no truncation; no image-count cap |
