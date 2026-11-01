# Submissions

| Model | Edition | Decision Index | Complete results | Engine / commit | Hardware | Declared limits |
|---|---|---:|---|---|---|---|
| EmbeddingGemma 2 System-One ([`Ajeet-001/embeddinggemma2-decision-router`](https://huggingface.co/Ajeet-001/embeddinggemma2-decision-router)) | 0.3 | 6.99 | [`runs/embeddinggemma2-text-0.3`](https://huggingface.co/datasets/Ajeet-001/embeddinggemma2-decision-index-results/tree/main/runs/embeddinggemma2-text-0.3) | `decision_index_engine:EmbeddingGemma2Engine` from the model repo (`temperature=0.05`, `null_threshold=0.58`), bf16; kit `25c6ec7` | 1 x NVIDIA A100-SXM4-40GB | `context_limit_tokens=2048` (strict no-truncation policy: inputs > 2,048 tokens raise `Unsupported`; 135,010 answered, 5,168 unsupported, 0 errors) |

| Board | Model | Hub repo | Results | Engine and options | Kit commit | Hardware | Resolution | Declared limits |
|---|---|---|---|---|---|---|---|---|
| **vision** | EmbeddingGemma 2 System-One | [Ajeet-001/embeddinggemma2-decision-router](https://huggingface.co/Ajeet-001/embeddinggemma2-decision-router) (public) | [runs/embeddinggemma2-vision-0.3/](https://huggingface.co/datasets/Ajeet-001/embeddinggemma2-decision-index-results/tree/main/runs/embeddinggemma2-vision-0.3) | `decision_index_engine:EmbeddingGemma2VisionEngine` from the model repo (`temperature=0.08`, `max_pixels=1605632`); bf16 | 25c6ec7 | 1 x NVIDIA A100-SXM4-40GB, bf16 | up to 1.6 MP (`max_pixels=1605632`), then `EmbeddingGemma2Processor` (384x384) | none; 12,210 / 12,210 rows answered, nothing truncated |
