# Submissions

| Model | Edition and public score | Complete results | Engine / commit and inference settings | Hardware | Declared limits |
|---|---|---|---|---|---|
| [Saina Helm 2.0](https://huggingface.co/run-saina/saina-helm-2-0.8b/tree/9903c487a3b1eef45c8cc56c76155b9baccb55f9) | Decision Index 0.3 public index: **30.17** (self-run; not an official leaderboard score) | [scores.json](https://huggingface.co/datasets/run-saina/saina-decision-index-results/blob/main/runs/9903c487a3b1eef45c8cc56c76155b9baccb55f9/scores.json); [reproducible evaluation repository](https://github.com/run-saina/saina-decision-index) | Saina saina.jev adapter 0.3.2 via saina_decision_index:SainaEngine; evaluation kit commit 9eb2dbe2a358004c8782c66e40a83ac07b953fec; harness commit 18ff7834f3ca88d1d224edf50cb87124385bf696; CUDA, max_length=262144, max_options=588, shared-trunk pass enabled, no truncation, no gold access, convolution kernel kernels-community/mamba-ssm@20b2508ad12ae40260291539bf45183000451850. Requested public model revision: 9903c487a3b1eef45c8cc56c76155b9baccb55f9. | One NVIDIA GeForce RTX 3090; PyTorch 2.14.1+cu130 | No capacity limits declared; all 140,178 requests completed. |

## Validation note

The published 0.3 run is complete (complete: true, 140,178/140,178 requests, public index 30.17). However, its environment.json records the loaded model as run-saina/saina-helm-2-0.8b-candidates at revision 378ba1e189bf646855ded08b737e02e8895a12da, rather than the submitted public model repository and revision above. The public model card and evaluation repository identify revision 9903c487a3b1eef45c8cc56c76155b9baccb55f9 with this score, but the run metadata does not establish that the measured checkpoint is identical. This discrepancy is disclosed for maintainer review.

The 30.17 value is only the self-run public 0.3 index. The official leaderboard Full score includes private tests and has not been assigned.
