# Decision Index submissions

<!-- rev-27b-midi-gen1-di-03 -->

## Midi — Decision Index 0.3

| Model | Results | Engine / commit | Hardware | Declared limits |
|---|---|---|---|---|
| [rev-27B-midi-gen1](https://huggingface.co/datasets/delip/rev-gen1-decision-index-021-20261005/blob/9f8dd9a1b8949804f9cc0047a69c553c6bed372b/checkpoints/rev-27b-midi-gen1/export-v1/export-manifest.json) | [Complete 0.3 public run: 51.59](https://huggingface.co/datasets/delip/rev-gen1-decision-index-021-20261005/blob/b6199faecbf694eb41e643857c6e300ede5a6dfc/runs/rev-27b-midi-gen1-0.3/scores.json) | [`midi_engine:MidiEngine`](https://huggingface.co/datasets/delip/rev-gen1-decision-index-021-20261005/blob/76b52247382863514bac304defc2d8ce392ba343/code/midi-20261008/inference-package.tar.gz); official runner `9eb2dbe2a358004c8782c66e40a83ac07b953fec` | One NVIDIA RTX PRO 6000 Blackwell, Modal ccb-lab/main; single process, one request at a time | 8,192 tokens per criterion path; 32,768 tokens per question tree; 255 options; no truncation or option filtering; 2,764 unsupported. Private artifacts; maintainer access required. |
