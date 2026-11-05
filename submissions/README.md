# Model submissions


<!-- xiangxin-3.0.7-20261103 -->
## Xiangxin 3.0.7

- Model: [xiangxinai/xiangxin-3.0.7](https://huggingface.co/xiangxinai/xiangxin-3.0.7/tree/3999b14d899dedd096e804fdf4f2813a360c8701) (private; maintainer access required).
- Complete public 0.3 run: [scores.json](https://huggingface.co/datasets/xiangxinai/xiangxin-3.0.7-decision-index/blob/813f9de2380ae4c3ae097bf9664da2f3bf1dff67/runs/public-0.3/scores.json), public index **57.47**; [predictions and reports](https://huggingface.co/datasets/xiangxinai/xiangxin-3.0.7-decision-index/tree/813f9de2380ae4c3ae097bf9664da2f3bf1dff67/runs/public-0.3).
- Kit: `f8552a13b643d39f61185f5951bb5fed16bedf51`; unmodified official scorers.
- Engine: direct production-backend HTTP `/v1/systemone`, checkpoint `xiangxin-3.0.7` (served alias `xiangxin-3.0`), concurrency 4 then 8; client batches independent questions on aggregate capacity rejection and merges original question IDs. Runner, batching adapter and pinned container ID included with results.
- Inference: vLLM 0.33.0 with Xiangxin commit `7674a81ee56909fcc19cf6707f1263c86e24a6db`; exact overlay and Docker build instructions in the private model repo's `inference/` directory.
- Hardware/settings: 2 × NVIDIA RTX A6000 48 GB, BF16, TP2, eager, context/batch 65,536 tokens, max 16 sequences, KV cache 2,700,000,000 bytes per GPU, GPU memory utilization 0.91.
- Temperatures: choice 1.5505, noul 1.3817; the original checkpoint configuration and readout are included. No input truncation, option filtering, second reading, or benchmark-specific prompt changes.
- Scoreable request statuses: `{"ok": 140178}`. Explicit capacity rejections count as wrong under the official scorer.
- Backend HTTP wall times include concurrency. Official single RTX PRO 6000 latency and private Full score remain for maintainers to measure.
