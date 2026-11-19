# Model submissions

| Model | Panel | Public index | Model revision | Results | Engine and hardware |
|---|---|---:|---|---|---|
| [SingVL S1 4B](https://huggingface.co/inclusionAI/SingVL_S1_4B) | Text | 69.75 | [dd9fcbe](https://huggingface.co/inclusionAI/SingVL_S1_4B/tree/dd9fcbe2412068def01666b1d1a443e412f11214) | [scores.json](https://huggingface.co/datasets/echoxvf/SingVL_S1_4B-DI-results/blob/709aaf417ebf1d00dd01fdb49e306573b801d488/releases/batch-65k/text/scores.json) | Transformers BF16, model-hosted `inference.engine:D3Engine`, 16 independent PPU workers |
| [SingVL S1 4B](https://huggingface.co/inclusionAI/SingVL_S1_4B) | Vision | 73.74949871316771 | [dd9fcbe](https://huggingface.co/inclusionAI/SingVL_S1_4B/tree/dd9fcbe2412068def01666b1d1a443e412f11214) | [vision-scores.json](https://huggingface.co/datasets/echoxvf/SingVL_S1_4B-DI-results/blob/709aaf417ebf1d00dd01fdb49e306573b801d488/releases/batch-65k/vision/vision-scores.json) | Transformers BF16, same native adapter and hardware |

Both panels use the pinned model's native `inference.engine:D3Engine` in BF16. For multi-question text requests, the engine batches complete prompts with a default budget of 65,536 padded tokens per forward pass. The public runs used [Decision Index kit commit `f5a6e4f`](https://github.com/apolinario/decision-index/tree/f5a6e4fd85fed05e77878dfc4966d0568bfb45ec), PyTorch 2.10.0 and Transformers 5.13.1.

Follow the [reproduction guide](https://huggingface.co/inclusionAI/SingVL_S1_4B/blob/c7a7f32ab027735b7dd2742e5b6cfcacaa43f97a/inference/DECISION_INDEX.md) to prepare the suites and run the Text and Vision panels. Use the same model revision and engine for private evaluation.
