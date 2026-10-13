# Submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| Amos 1 large ([`mukaddamzaid/amos-1-large`](https://huggingface.co/mukaddamzaid/amos-1-large) @ `362f4dc`) | 0.3 | 64.82 | [`runs/amos-1-large-0.3`](https://huggingface.co/datasets/mukaddamzaid/amos-1-decision-index/tree/255ac2e5e3ce3b1841022714f53cb531b5d96f9b/runs/amos-1-large-0.3) | `http` against `amos.serve` `/v1/systemone` (the repo's `serve/`), kit `9eb2dbe` | 4 × H100 80GB, one server per shard, one request at a time | 131,072 tokens per request; longer requests refused (422), never truncated; no options removed |
