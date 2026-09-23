# Submissions

| Model | Decision Index | Results | Engine and exact code | Hardware | Declared capacity limits |
|---|---:|---|---|---|---|
| GLiDE 27B (`Qwen/Qwen3.8-27B` weights, unmodified, bf16) | **62.29** | [full run and scores](https://huggingface.co/datasets/Sahibzada-a/decision-index-results/blob/main/runs/glide-27b/scores.json) | Batched vLLM engine with a letter-slot readout: next-token distribution restricted to the option letters. Two modes averaged 50/50 per question: hidden reasoning of up to 1,024 greedy tokens then readout, and a one-pass readout averaged over forward and reversed option order. One prompt for every benchmark. Kit `52a6989`; settings and file hashes in `runs/glide-27b/environment.json`. | NVIDIA H200 (Modal), one model replica per GPU | Choice questions up to 255 options; context 40,000 tokens. Nothing was truncated and no options were removed. All 132,422 requests returned `ok`. |
