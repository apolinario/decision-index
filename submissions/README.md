# Submissions

| Model | Decision Index | Results | Engine and exact code | Hardware | Declared capacity limits |
|---|---:|---|---|---|---|
| Bespoke-Nimble-9B-v3 (`bespokelabs/Bespoke-Nimble-9B-v3` @ `8e927b9b`, LoRA adapter on `Qwen/Qwen3.5-9B` @ `c2022362`, bf16) | **56.88** | [full run and scores](https://huggingface.co/datasets/bespokelabs/bespoke-nimble-9b-v3-decision-index/blob/654b124d3508a7da0d849405744bde80c4bb1a1b/runs/bespoke-nimble-9b-v3/scores.json) | `nimble.evaluation.decision_index_engine:NimbleEngine` (in the results dataset under `code/`), kit `87d4650` (edition 0.2.1), default options (`shared_prefix=true`) | 8 x NVIDIA H100 (80 GB); 16 runner processes (2 per GPU) | Choice questions up to 255 options; context 32,768 tokens. Nothing was truncated and no options were removed. 150,305 of 150,317 requests returned `ok`; 12 prompts over 32,768 tokens are `unsupported`. |
