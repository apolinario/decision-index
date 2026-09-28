# Submissions

| Model | Decision Index | Results | Engine and exact code | Hardware | Declared capacity limits |
|---|---:|---|---|---|---|
| OneJev-0.8B (`OmniJev/OneJev-0.8B` @ `3368dda`) | **22.03** | [full run and scores](https://huggingface.co/datasets/OmniJev/onejev-decision-index-results/blob/f0af61643d1765a202abf3524d67370b19f80a79/runs/onejev-0.8b/scores.json) | `decision_index_engine.QevEngine` (one prefill per request, one branch per question, answer read off the label logits; debias 1, float32 head) in a kit-format engine; [OneJev](https://github.com/OmniJev/OneJev) `9f4a42e` (`benchmarks/decision_index_engine.py`); kit `87d4650` (0.2.1 scoring) | NVIDIA H200, 8 shard jobs, bf16 | None: 150,317 of 150,317 scoreable requests answered; nothing truncated (branch limit raised to 131,072 tokens). |
