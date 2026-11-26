# Model submissions

| Model | Boards / public index | Results | Inference code | Hardware | Inference settings |
| --- | --- | --- | --- | --- | --- |
| [Owl-Jev](https://huggingface.co/Jaysonddd/owl-jev-27b/tree/938f9b4880e06f5c69ba5f57458084e483f879e1) | Text 68.22 / Vision 80.38 (Public 0.3) | [Text scores](https://huggingface.co/datasets/Jaysonddd/owl-jev-results/blob/ddf5a44ea46cb2b222adaf037ee7501be7c2bd5c/runs/owl-jev-text/scores.json), [Vision scores](https://huggingface.co/datasets/Jaysonddd/owl-jev-results/blob/ddf5a44ea46cb2b222adaf037ee7501be7c2bd5c/runs/owl-jev-vision/vision-scores.json), [Vision run record](https://huggingface.co/datasets/Jaysonddd/owl-jev-results/blob/ddf5a44ea46cb2b222adaf037ee7501be7c2bd5c/runs/owl-jev-vision/environment.json) | `owl_jev.engine:TextEngine`, `owl_jev.engine:VisionEngine`. [Reproduction instructions](https://huggingface.co/datasets/Jaysonddd/owl-jev-results/blob/ddf5a44ea46cb2b222adaf037ee7501be7c2bd5c/README.md#run), kit `25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d` | 8 × 8 NVIDIA A800 GPUs | BF16 backbone, FP32 readout, SDPA, noncausal attention, last-token pooling, temperature 1.0, batch size 1, no permutation averaging. Vision uses 589,824–1,638,400 pixels per image. `max_length=null`. |

Text covers 140,178 scored requests. Vision covers 12,210 requests scored from archived predictions.
