# Submissions

| Board | Model | Hub repo | Results | Engine and options | Kit commit | Hardware | Resolution | Declared limits |
|---|---|---|---|---|---|---|---|---|
| **vision** | Aster-JEV-27B | [Model](https://huggingface.co/k2021211255/Aster-JEV-27B/tree/574f6a9e37909c88bf485c521bfb7cf359ca9e1a) | [Complete run](https://huggingface.co/datasets/k2021211255/Aster-JEV-27B-Results/tree/9d9f28da6a077380bbbcca430065a54e668bf9d2/runs/Aster-JEV-27B) | `aster_jev.engine:AsterJEVEngine`; `model=k2021211255/Aster-JEV-27B@6534b9903284dce412bc9065afc9d6ab6b5724cf` | `25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d` | A800 80GB; BF16 | Native up to 1,638,400 pixels per image; lower author-set caps lifted | Choice only; 2–26 options; 262,144 tokens; no image-count cap; explicit capacity errors |
