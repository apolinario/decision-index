# Model submissions

| Model | Panel | Public index | Model revision | Results | Engine and hardware |
|---|---|---:|---|---|---|
| [SingVL S1 4B](https://huggingface.co/inclusionAI/SingVL_S1_4B) | Text | 69.75 | [dd9fcbe](https://huggingface.co/inclusionAI/SingVL_S1_4B/tree/dd9fcbe2412068def01666b1d1a443e412f11214) | [scores.json](https://huggingface.co/datasets/echoxvf/SingVL_S1_4B-DI-results/blob/709aaf417ebf1d00dd01fdb49e306573b801d488/releases/batch-65k/text/scores.json) | Transformers BF16, model-hosted `inference.engine:D3Engine`, 16 independent PPU workers |
| [SingVL S1 4B](https://huggingface.co/inclusionAI/SingVL_S1_4B) | Vision | 73.74949871316771 | [dd9fcbe](https://huggingface.co/inclusionAI/SingVL_S1_4B/tree/dd9fcbe2412068def01666b1d1a443e412f11214) | [vision-scores.json](https://huggingface.co/datasets/echoxvf/SingVL_S1_4B-DI-results/blob/709aaf417ebf1d00dd01fdb49e306573b801d488/releases/batch-65k/vision/vision-scores.json) | Transformers BF16, same native adapter and hardware |

Use the [pinned model and adapter](https://huggingface.co/inclusionAI/SingVL_S1_4B/tree/dd9fcbe2412068def01666b1d1a443e412f11214) for both public and private Text/Vision evaluation. The [batched release results](https://huggingface.co/datasets/echoxvf/SingVL_S1_4B-DI-results/tree/709aaf417ebf1d00dd01fdb49e306573b801d488/releases/batch-65k) contain predictions and official scorer outputs. Evaluation used Decision Index kit commit `f5a6e4fd85fed05e77878dfc4966d0568bfb45ec`, PyTorch 2.10.0 and Transformers 5.13.1.

Within a multi-question text request, each question has its own complete prompt. The predictor batches these prompts by length with at most 65,536 padded tokens per forward pass. Single-question text and image requests use the serial path. Full attention is prefix-bidirectional, GDN is causal and KV caching is disabled.

Limits are four images per request, 256 choice candidates, 65,536 input tokens per text prompt and 10,000 tokens for prompts with images. Images use aspect-preserving, patch-grid-aligned RGB bicubic resizing capped at 1,638,400 pixels per image. Recorded maxima are 1,638,400 pixels per image and 3,200 image tokens per request. Private Text/Vision evaluation and the official single-RTX-PRO-6000 latency measurement are requested.

The weights match the [original checkpoint](https://huggingface.co/inclusionAI/SingVL_S1_4B/tree/93f15cd61c1cf417d9cb2631f6be2df13510829e). The [serial adapter](https://huggingface.co/inclusionAI/SingVL_S1_4B/tree/b6672ce4fd31aab79aee3f0729fd8738f87361cf) and the [original submission snapshot](https://huggingface.co/datasets/echoxvf/SingVL_S1_4B-DI-results/tree/90ec7e159da6fa21d762f1f75876ea79049d0f4a) document the prior run (Text 69.76 and Vision 73.74949871316771).

[Decision Index reproduction guide](https://huggingface.co/inclusionAI/SingVL_S1_4B/blob/dd9fcbe2412068def01666b1d1a443e412f11214/inference/DECISION_INDEX.md) · [Native adapter](https://huggingface.co/inclusionAI/SingVL_S1_4B/blob/dd9fcbe2412068def01666b1d1a443e412f11214/inference/engine.py)
