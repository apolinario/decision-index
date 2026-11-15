# Submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| [Nischay-1](https://huggingface.co/Ajeet-001/nischay-1/tree/d201de0d82b233c6ffa041133378e0e673254450) | 0.3 | 50.81 | [scores.json](https://huggingface.co/datasets/Ajeet-001/nischay-1-text-decision-index-results/blob/fe8a5c6eb69abab85098ddf9aca1318569cab1a0/runs/nischay-1/scores.json) | `engine:DecisionModel` (`dtype=bfloat16`) @ `d201de0`; kit `25c6ec7` | 1 x NVIDIA A100-SXM4-40GB | 65,536 branch / 131,072 request tokens; 0 unsupported |

| Board | Model | Hub repo | Results | Engine and options | Kit commit | Hardware | Resolution | Declared limits |
|---|---|---|---|---|---|---|---|---|
| **vision** | Nischay-1 | [Ajeet-001/nischay-1](https://huggingface.co/Ajeet-001/nischay-1/tree/d201de0d82b233c6ffa041133378e0e673254450) | [vision-scores.json](https://huggingface.co/datasets/Ajeet-001/nischay-1-vision-decision-index-results/blob/f1884d2be58c891c5bf24bae7f791857645e5150/runs/vision-nischay-1/vision-scores.json) (53.57, 12,210/12,210) | `engine:DecisionModel` (`dtype=bfloat16`, `max_pixels=1638400`, `max_soft_tokens=560`) @ `d201de0` | `25c6ec7` | 1 x NVIDIA A100-SXM4-40GB | Native up to 1,638,400 px, Gemma 4 `max_soft_tokens=560` | 8,192 tokens; 0 unsupported |
