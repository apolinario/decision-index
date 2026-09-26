# Submissions

| Model | Results | Engine / source | Hardware |
|---|---|---|---|
| Reflex 27B (Qwen3.8-27B-FP8, two orderings) | [Complete Decision Index 0.2 run](https://huggingface.co/datasets/Kshetrajna/reflex-27b-h100-decision-index-02/blob/d2366034a995309a6552cdf7d7dd54d23133479b/runs/reflex-27b-h100-di02/scores.json) | Reflex base `d08a8c6cd63a81164c27f4c0f1f10b991eb2c32d` with [the exact serving changes](https://huggingface.co/datasets/Kshetrajna/reflex-27b-h100-decision-index-02/blob/d2366034a995309a6552cdf7d7dd54d23133479b/serving-source/source-lock.json), SGLang | 1 × H100 80GB; 32 HTTP workers |

Reflex scored 47.83 with all 151,476 requests successful (151,034 after the standard exclusions). The context setting is 65,536 tokens; choice limits depend on the tokenizer's one-token labels. Every question in the suite was supported without truncation or option filtering. The [dataset](https://huggingface.co/datasets/Kshetrajna/reflex-27b-h100-decision-index-02/tree/d2366034a995309a6552cdf7d7dd54d23133479b) includes predictions, serving source and runtime details.
