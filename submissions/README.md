# Submissions

| Board | Model | Hub repo | Results | Engine and options | Kit commit | Hardware | Resolution | Declared limits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **vision** | JDI Decision Flash V1 (9B) | [Model](https://huggingface.co/JoyIndustrial/JDI-decision-flash-v1/tree/af0c00a9bc02793c71a44770d903f0b6c888d4b4) · [Reproduction](https://huggingface.co/JoyIndustrial/JDI-decision-flash-v1/blob/af0c00a9bc02793c71a44770d903f0b6c888d4b4/REPRODUCE.md) | [Run](https://huggingface.co/datasets/JoyIndustrial/JDI-decision-vision-results/tree/05ad6cb856035d223712c7700bd0ecec5052c27f/runs/flash) (72.68, 12,210/12,210) | `official_engine:JDIVision`, `variant=flash`, `ocr_device=gpu:0`, PP-OCRv6 medium | `25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d` | 2 × NVIDIA H200, one replica and one request per GPU | Native up to 1,638,400 pixels per image, measured maximum 1,638,400 | Multiple-choice, one question per request, two images at most observed |
