# Submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| Amos 1 large ([`mukaddamzaid/amos-1-large`](https://huggingface.co/mukaddamzaid/amos-1-large) @ `362f4dc`) | 0.3 | 64.94 | [`runs/amos-1-large-0.3`](https://huggingface.co/datasets/mukaddamzaid/amos-1-decision-index-results/tree/8aec0e2d21cb33a48d747e15d05aa7fe65a2aa48/runs/amos-1-large-0.3) | `http` against `amos.serve` `/v1/systemone` (the repo's `serve/`), `--rotations 2`, kit `9eb2dbe` ([how to serve](Amos-1-large/README.md)) | 4 × H100 80GB, one server per shard, one request at a time | 131,072 tokens per request; longer requests refused (422), never truncated; no options removed |
