# Model submissions

| Model | Model repository | Complete public results | Engine / hardware / settings |
|---|---|---|---|
| Aplomb 2 | [empiriolabsai/aplomb-2](https://huggingface.co/empiriolabsai/aplomb-2/tree/399027c60602cd832724456b82d24d467ed9cad0) | [Decision Index 0.3 scores.json](https://huggingface.co/datasets/empiriolabsai/aplomb-2-decision-index/blob/b375e8344afc2989526fd1b3d68535321a38dbe2/runs/aplomb-2/scores.json) | Kit `http` engine against the model's `/v1/systemone` server (`serve_aplomb.py` in the model repository); kit `9eb2dbe2a358004c8782c66e40a83ac07b953fec`; RTX PRO 6000 Blackwell 96GB; BF16; one request at a time for latency; [settings](aplomb-2/README.md). |
