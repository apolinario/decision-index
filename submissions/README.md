# Submissions

| Board | Model | Hub repo | Results | Engine and options | Kit commit | Hardware | Resolution | Declared limits |
|---|---|---|---|---|---|---|---|---|
| **vision** | Sieve-Vision-4B | [sthanika-ai/Sieve-Vision-4B](https://huggingface.co/sthanika-ai/Sieve-Vision-4B/tree/fdc8954555beb7c73e3fc3e2919512e713fb84ee) | [runs/Sieve-Vision-4B](https://huggingface.co/datasets/sthanika-ai/Sieve-Vision-4B-decision-index-results/tree/dd8ac232ace5ef6ce7c8af5c284b129bf493652c/runs/Sieve-Vision-4B) | [`sieve_vision_engine:SieveVisionEngine`](https://huggingface.co/sthanika-ai/Sieve-Vision-4B/blob/fdc8954555beb7c73e3fc3e2919512e713fb84ee/sieve_vision/engine.py) (in the model repo), `model=sthanika-ai/Sieve-Vision-4B` | `f5a6e4f` | 1× A100 80GB PCIe, bf16 | native up to 1.6 MP per image (no lower cap) | none: any number of images and options; nothing truncated |
