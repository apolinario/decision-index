# Submissions

| Board | Model | Hub repo | Results | Engine and options | Kit commit | Hardware | Resolution | Declared limits |
|---|---|---|---|---|---|---|---|---|
| **vision** | Aster-JEV-27B | [Model](https://huggingface.co/k2021211255/Aster-JEV-27B/tree/a7058b6ddc7229785ccd4b036543dd450b1d38fb) | [Complete run](https://huggingface.co/datasets/k2021211255/Aster-JEV-27B-Results/tree/404d123a4bcb0694a9d1cf0666f44df919df61da/runs/Aster-JEV-27B) | `aster_jev.engine:AsterJEVEngine`; `model=k2021211255/Aster-JEV-27B@e76cfd64e30929a2c82992bfbb467f9e1e89767b` | `25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d` | A800 80GB; BF16 | Native up to 1,638,400 pixels per image; lower author-set caps lifted | Choice only; 2–26 options; 262,144 tokens; no image-count cap; explicit capacity errors |
