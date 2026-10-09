# Submissions

| model | model repo | results | engine / code | hardware | settings |
|---|---|---|---|---|---|
| linne-fern-1.0 | [MISA-core/linne-fern-1-0](https://huggingface.co/MISA-core/linne-fern-1-0) @ 5b73573 (weights) | [MISA-core/linne-fern-1-0-decision-index/runs/linne-fern-1-0](https://huggingface.co/datasets/MISA-core/linne-fern-1-0-decision-index/tree/main/runs/linne-fern-1-0) (public 0.3: 58.92, complete) | kit 0.3 (62d2f51), in-process `linne_engine:LinneEngine` (`linne_engine.py` in the model repo) | 4x H200 (8 processes, 2 per GPU) | bf16, SDPA; one forward pass per question, no thinking, no generated tokens; softmax (temperature 1) over the option-label logits; token budget 12288 per batch, max length 65536; nothing truncated; 0 errors, 0 unsupported |
