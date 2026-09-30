# Submissions

| Model | Decision Index | Results | Engine and exact code | Hardware | Declared capacity limits |
|---|---:|---|---|---|---|
| Nimble V3 (`bespokelabs/Nimble-V3` @ `253de670`, LoRA adapter on `Qwen/Qwen3.5-9B` @ `c2022362`, bf16) | **56.88** | [full run and scores](https://huggingface.co/datasets/bespokelabs/nimble-v3-decision-index/blob/6ace2a44ad4cc0c09c59d9bc0cf6511aa4a3cd5b/runs/nimble-v3/scores.json) | `nimble.evaluation.decision_index_engine:NimbleEngine` (in the results dataset under `code/`), kit `87d4650` (edition 0.2.1), default options (`shared_prefix=true`) | 8 x NVIDIA H100 (80 GB); 16 runner processes (2 per GPU) | Choice questions up to 255 options; context 32,768 tokens. Nothing was truncated and no options were removed. 150,305 of 150,317 requests returned `ok`; 12 prompts over 32,768 tokens are `unsupported`. |
