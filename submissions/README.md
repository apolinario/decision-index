# Decision Index submissions

<!-- rev-gen1-midi-lite-di-03 -->

## rev-gen1-midi-lite — Decision Index 0.3

| Model | Results | Engine / commit | Hardware | Declared limits |
|---|---|---|---|---|
| [rev-gen1-midi-lite](https://huggingface.co/datasets/delip/rev-gen1-decision-index-021-20261005/blob/4a09260456fa15ba3991bcedc94d5d7f3c7ef3cf/checkpoints/rev-gen1-midi-lite/export-v1/export-manifest.json) | [Complete 0.3 public run: 48.41](https://huggingface.co/datasets/delip/rev-gen1-decision-index-021-20261005/blob/52ce5788d5705226a2e93b5e0fbb436048495ee1/runs/rev-gen1-midi-lite-0.3/scores.json) | [`lite_engine:LiteEngine`](https://huggingface.co/datasets/delip/rev-gen1-decision-index-021-20261005/blob/25b81432c67a19f0c075bd62de922b7a87807dcb/code/lite-20261010/rev-gen1-midi-lite/inference-package.tar.gz); `REV_MODEL_KEY=rev-gen1-midi-lite`; official runner `9eb2dbe2a358004c8782c66e40a83ac07b953fec` | One NVIDIA RTX PRO 6000 Blackwell, Modal ccb-lab/main; single process, one request at a time | 8,192 tokens per criterion path; 32,768 tokens per question tree; 255 options; no truncation or option filtering; 2,764 unsupported. Private artifacts; maintainer access required. |
