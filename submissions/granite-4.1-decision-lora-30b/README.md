# Granite 4.1 Decision LoRA · Granite-4.1-30B evaluation harness

IBM's open [Decision LoRA for Granite-4.1-30B](https://huggingface.co/ibm-granite/granitelib-core-r1.0/tree/main/decision/granite-4.1-30b/lora) ([`ibm-granite/granitelib-core-r1.0`](https://huggingface.co/ibm-granite/granitelib-core-r1.0), adapter `decision/granite-4.1-30b/lora`; Apache-2.0) is a standard PEFT LoRA (r=64, α=128) on frozen [`ibm-granite/granite-4.1-30b`](https://huggingface.co/ibm-granite/granite-4.1-30b), never merged. It reads the state once and scores every supplied option in one forward pass (`max_tokens=1`, `temperature=0`), softmax over single-token option-label logits; it generates nothing.

> **Decision Index 0.3 (public) = 54.04.**
> Per area (×100): knowledge 36.9, language 61.5, retrieval 63.9, tools 71.4, arts 27.9.

The engine is the kit's logit-readout against stock vLLM serving base + adapter (`--enable-lora --lora-modules decision-lora=<adapter>`, not merged, not the aLoRA fork), with the model-card prompt: one user turn, no system message, lettered options `A: desc`, `add_generation_prompt=True`.

## Run

One 80 GB GPU. Prereqs: a Python env with `vllm>=0.30`, `transformers>=4.44`, `huggingface_hub`, `httpx`, `numpy`, `git`, and `huggingface-cli login` (or `HF_TOKEN`) with access to the gated suite dataset.

```sh
# fetch the eval script + decider from the eval-scripts-granite-4-1 branch of this fork, then:
bash eval_granite_decision_lora.sh
```

[`eval_granite_decision_lora.sh`](https://github.com/subhajitchaudhury/decision-index/blob/eval-scripts-granite-4-1/submissions/eval-scripts-granite-4-1/eval_granite_decision_lora.sh) fetches everything else (the `decision-index` kit, the frozen suite, the base [`ibm-granite/granite-4.1-30b`](https://huggingface.co/ibm-granite/granite-4.1-30b), the adapter), serves base + adapter on stock vLLM, runs the decider ([`decider_granite_decision_lora.py`](https://github.com/subhajitchaudhury/decision-index/blob/eval-scripts-granite-4-1/submissions/eval-scripts-granite-4-1/decider_granite_decision_lora.py)), scores at edition 0.3.2, and prints the Decision Index + per-area skills.

The eval script, decider, and the produced [`scores.json`](https://github.com/subhajitchaudhury/decision-index/blob/eval-scripts-granite-4-1/submissions/eval-scripts-granite-4-1/scores.json) live on the [`eval-scripts-granite-4-1`](https://github.com/subhajitchaudhury/decision-index/tree/eval-scripts-granite-4-1/submissions/eval-scripts-granite-4-1) branch of this fork.

