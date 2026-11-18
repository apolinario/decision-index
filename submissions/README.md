# Model submissions

| Model | Panel | Public index | Model revision | Results | Engine and hardware |
|---|---|---:|---|---|---|
| [SingVL S1 4B](https://huggingface.co/inclusionAI/SingVL_S1_4B) | Text | 69.76 | [b6672ce](https://huggingface.co/inclusionAI/SingVL_S1_4B/tree/b6672ce4fd31aab79aee3f0729fd8738f87361cf) | [scores.json](https://huggingface.co/datasets/echoxvf/SingVL_S1_4B-DI-results/blob/90ec7e159da6fa21d762f1f75876ea79049d0f4a/runs/singvl-s1-4b-text/scores.json) | Transformers BF16, native `inference.engine:D3Engine` from model `inference/`, 16 independent PPU workers |
| [SingVL S1 4B](https://huggingface.co/inclusionAI/SingVL_S1_4B) | Vision | 73.74949871316771 | [b6672ce](https://huggingface.co/inclusionAI/SingVL_S1_4B/tree/b6672ce4fd31aab79aee3f0729fd8738f87361cf) | [vision-scores.json](https://huggingface.co/datasets/echoxvf/SingVL_S1_4B-DI-results/blob/90ec7e159da6fa21d762f1f75876ea79049d0f4a/runs/singvl-s1-4b-vision/vision-scores.json) | Transformers BF16, native `inference.engine:D3Engine` from model `inference/`, 16 independent PPU workers |

The kit commit is `f5a6e4fd85fed05e77878dfc4966d0568bfb45ec`. Results and adapter revision are `90ec7e159da6fa21d762f1f75876ea79049d0f4a` in [echoxvf/SingVL_S1_4B-DI-results](https://huggingface.co/datasets/echoxvf/SingVL_S1_4B-DI-results). The runtime uses PyTorch 2.10.0, Transformers 5.13.1, prefix-bidirectional full attention, causal GDN, one complete forward pass per question and disabled KV caching.

Limits are four images, 256 choice candidates, 65536 text input tokens and 10000 tokens for image inputs. Image preprocessing uses native aspect-preserving, patch-grid-aligned RGB bicubic resizing with a 1638400-pixel cap. Recorded maxima are 1638400 pixels per image and 3200 image tokens per request. Private Text/Vision evaluation and the official single-RTX-PRO-6000 latency measurement are requested.

The downloadable model and complete DI adapter are pinned to `b6672ce4fd31aab79aee3f0729fd8738f87361cf`. The weights and native predictor are unchanged from the originally evaluated model revision `93f15cd61c1cf417d9cb2631f6be2df13510829e`. The original results and code snapshot remain at dataset revision `90ec7e159da6fa21d762f1f75876ea79049d0f4a`.

Download, installation, native Text/Vision evaluation and HTTP startup commands: [Decision Index reproduction guide](https://huggingface.co/inclusionAI/SingVL_S1_4B/blob/main/inference/DECISION_INDEX.md). The adapter is also available directly at [inference/engine.py](https://huggingface.co/inclusionAI/SingVL_S1_4B/blob/b6672ce4fd31aab79aee3f0729fd8738f87361cf/inference/engine.py).
