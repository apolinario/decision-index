# Submissions

| Model | Edition and public score | Complete results | Engine / commit and inference settings | Hardware | Declared limits |
|---|---|---|---|---|---|
| [Saina Helm 2.0](https://huggingface.co/run-saina/saina-helm-2-0.8b/tree/06f2fec2be4355b009b323f248a26eecca52606a) | Decision Index 0.3 public index: **30.47** (self-run; not an official leaderboard score), from model revision 06f2fec2be4355b009b323f248a26eecca52606a | [scores.json](https://huggingface.co/datasets/run-saina/saina-decision-index-results/blob/main/runs/06f2fec2be4355b009b323f248a26eecca52606a/scores.json); [reproducible evaluation repository](https://github.com/run-saina/saina-decision-index) | Saina saina.jev adapter 0.3.2 via saina_decision_index:SainaEngine; evaluation kit commit 9eb2dbe2a358004c8782c66e40a83ac07b953fec; harness commit 18ff7834f3ca88d1d224edf50cb87124385bf696; CUDA, max_length=262144, max_options=588, shared-trunk pass enabled, no truncation, no gold access, convolution kernel kernels-community/mamba-ssm@20b2508ad12ae40260291539bf45183000451850. | One NVIDIA GeForce RTX 3090; PyTorch 2.14.1+cu130 | No capacity limits declared; all 140,178 requests completed. |

The 30.47 value is the self-run public 0.3 index only. The official leaderboard Full score includes private tests and has not been assigned.
