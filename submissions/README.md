# Decision Index submissions

| Model | Results | Engine / commit | Hardware | Declared limits |
|---|---|---|---|---|
| Rev-2B-micro-gen1 | [Complete 0.2.1 run](https://huggingface.co/datasets/delip/rev-gen1-decision-index-021-20261005/blob/e4bed25b82663be598fa8dab4773ef0956ede581/runs/rev-2b-micro-gen1/scores.json) | [`rev_engine:RevEngine`](https://huggingface.co/datasets/delip/rev-gen1-decision-index-021-20261005/blob/e4bed25b82663be598fa8dab4773ef0956ede581/code/inference-package.tar.gz); official runner `87d4650b42b377c0291a89c1f1a879f9b31082bf`; frozen source manifest `a36f2931523559acf1d2cdf9aaad43cb6dcd0a08a8f18d55eac677b64833fc96` | One NVIDIA RTX PRO 6000 Blackwell Server Edition, Modal ccb-lab/main; single process, one request at a time | 8,192 tokens per complete criterion path; 32,768 tokens per question tree; 255 criteria per question. Unrepresentable frozen-renderer inputs are unsupported; no truncation or option filtering. |
