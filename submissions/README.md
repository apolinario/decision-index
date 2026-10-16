# Submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| elephant-1 ([`tusk-force/elephant-1`](https://huggingface.co/tusk-force/elephant-1) @ `9ac8f17`) | 0.3 | 69.81 | [`runs/elephant-1`](https://huggingface.co/datasets/tusk-force/elephant-1-decision-index/tree/896432a310999b105046be176a8c41c485d5e4ad/runs/elephant-1) | in-process `engine:DecisionModel` (the repo's `engine.py`, OneJev `81ce62f` + `serve.patch`), kit `f5a6e4f` | 2 × 8 H100 80GB, one engine per GPU, one request at a time | 65,536 tokens per request; longer requests refused, never truncated; no options removed |
