# Submissions

| Board | Model | Hub repo | Results | Engine and options | Kit commit | Hardware | Resolution | Declared limits |
|---|---|---|---|---|---|---|---|---|
| **vision** | Aster-JEV-27B | [k2021211255/Aster-JEV-27B](https://huggingface.co/k2021211255/Aster-JEV-27B/tree/cc8653ae6bfec8ed831c11d7370e0b3cf54c4427) | [Complete run](https://huggingface.co/datasets/k2021211255/Aster-JEV-27B-Results/tree/4758b79902a052bbc5a02eb9385cf2c56901167b/runs/Aster-JEV-27B) | [`aster_jev.engine:AsterJEVEngine`](https://huggingface.co/k2021211255/Aster-JEV-27B/blob/cc8653ae6bfec8ed831c11d7370e0b3cf54c4427/aster_jev/engine.py); `model=k2021211255/Aster-JEV-27B@cc8653ae6bfec8ed831c11d7370e0b3cf54c4427` | `25c6ec7` | 8× A800 80GB, independent serial single-GPU engines; BF16 | native up to 1,638,400 pixels per image; no shared or lower author cap | choice only, 2–26 options, 262,144 input tokens; no image-count cap; capacity failures reported |
