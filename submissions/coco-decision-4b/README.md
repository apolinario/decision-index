---
license: cc-by-nc-4.0
base_model:
  - Qwen/Qwen3.5-4B
  - corners-ai/CoCo-Decision-4B-Ko
base_model_relation: finetune
library_name: transformers
pipeline_tag: text-classification
language:
  - en
  - ko
tags:
  - decision-model
  - calibration
  - system-one
  - decision-index
model-index:
  - name: CoCo-Decision-4B
    results:
      - task:
          type: text-classification
          name: Decision (System One)
        dataset:
          name: Jev Decision Index 0.3 (public panel, self-scored)
          type: decision-index
        metrics:
          - type: decision_index
            value: 55.38
            name: Public index (balanced skill)
---

# CoCo-Decision-4B

**Version 2.3.0** (2026-10-12) · Git tag `v2.3.0` in this repository. Versions 2.2.0, 2.1.0, 2.0.0 and 1.0.0 stay available under tags `v2.2.0`, `v2.1.0`, `v2.0.0` and `v1.0.0`.

CoCo-Decision-4B is a 4B-class decision model. Given a situation and a typed question, it returns a probability distribution instead of free text:

| Question type | Output |
|---|---|
| `noul` | probability that a yes/no statement holds |
| `choice` | probability for each of the listed options |
| `score` | probability for each point on an ordinal scale |

It is the benchmark-oriented sibling of [corners-ai/CoCo-Decision-4B-Ko](https://huggingface.co/corners-ai/CoCo-Decision-4B-Ko): starting from CoCo-Decision-4B-Ko v1.0.0 merged into [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (revision `851bf6e8`), a second LoRA was trained only on the public train splits of the source datasets used by the [Jev Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index), then merged (version 1.0.0). Version 2.0.0 added a third LoRA, trained on top of version 1.0.0 with train splits of further Decision Index source datasets, then merged. Version 2.1.0 is a different third LoRA on top of version 1.0.0, merged, with a changed readout (see How to use). Versions 2.2.0 and 2.3.0 are retrained versions of that LoRA with the same structure. This repository holds the merged, text-only bf16 weights (about 4.2B parameters, no vision tower), so no PEFT is needed.

If you need Korean intent classification, use CoCo-Decision-4B-Ko. This model was not trained further on Korean data, and its Korean performance has not been re-measured.

> Not affiliated with or endorsed by TypeSafe AI. "System One" and "Jev" are names used by TypeSafe AI; this model independently implements a compatible decision interface.

## How to use

Requirements: `torch`, `transformers>=5.17`, `safetensors`, `huggingface_hub`.

Versions 2.1.0 and later do not read the answer from the language-model head. The full-attention layers attend over the whole prompt (no causal mask; padding stays masked), and the answer is read from the last hidden state with a small linear head (`readout.safetensors`) over the option-letter tokens listed in `decision_config.json`. The snippet below reproduces this with plain `transformers`; versions 2.0.0 and 1.0.0 use the language-model head as described under their tags.

```python
import json

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO, REVISION = "corners-ai/CoCo-Decision-4B", "v2.3.0"
path = snapshot_download(REPO, revision=REVISION)
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", dtype=torch.bfloat16).eval()
config = json.load(open(f"{path}/decision_config.json"))
table = config["readout_token_ids"]  # token ids of the option letters, one readout row each
readout = load_file(f"{path}/readout.safetensors")["proj.weight"].to(model.device, torch.float32)

# Full-attention layers read the whole prompt (no causal mask); padding stays masked.
inner = model.model
state = {}
inner.register_forward_pre_hook(lambda m, a, kw: state.update(mask=kw.get("attention_mask")), with_kwargs=True)


def open_mask(module, args, kwargs):
    hidden = args[0] if args else kwargs["hidden_states"]
    pad = state["mask"]
    if pad is None:
        pad = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
    kwargs["attention_mask"] = pad[:, None, None, :].bool()
    return args, kwargs


for kind, layer in zip(inner.config.layer_types, inner.layers):
    if kind == "full_attention":
        layer.register_forward_pre_hook(open_mask, with_kwargs=True)


def decide(state_text: str, question: str, options: dict[str, str]) -> dict[str, float]:
    """Return a probability for each option key."""
    letters = [chr(ord("A") + i) for i in range(len(options))]
    lines = ["State:", state_text, "", f"Question: {question}", "Options:"]
    lines += [f"{letter}. {key}: {text}" for letter, (key, text) in zip(letters, options.items())]
    lines.append("Answer with the letter only.")
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "\n".join(lines)}], tokenize=False, add_generation_prompt=True
    ) + "Answer:"
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        hidden = inner(**ids, use_cache=False).last_hidden_state[0, -1].float()
    rows = [table.index(tokenizer.encode(" " + letter, add_special_tokens=False)[0]) for letter in letters]
    probs = torch.softmax((readout @ hidden)[rows], dim=-1).tolist()
    return dict(zip(options, probs))
```

The prompt format is the same as in earlier versions: `State:`, `Question:`, `Options:` with one `<letter>. <key>: <description>` line per option, `Answer with the letter only.`, the chat template with `add_generation_prompt=True`, then `Answer:`. The probabilities are the softmax over the readout rows of the letters with a leading space.

Hardware: about 10 GB of GPU memory in bf16.

Benchmarking and side-by-side comparison with [oh-my-jev (omj)](https://github.com/iamupd/oh-my-jev), which also serves the model behind a TypeSafe-compatible endpoint:

```bash
git clone https://github.com/iamupd/oh-my-jev && cd oh-my-jev
uv sync --extra semif
uv run omj bench --model corners-ai/CoCo-Decision-4B --suite jevbench-public --view
uv run omj ui --target corners-ai/CoCo-Decision-4B --target corners-ai/CoCo-Decision-4B-Ko
```

## Training

- Starting point: CoCo-Decision-4B-Ko v1.0.0 (LoRA on Qwen/Qwen3.5-4B, see its card for the first-stage data) merged into the base model
- Second stage (version 1.0.0): LoRA on the train splits of 14 Decision Index source datasets, merged
- Third stage (version 2.0.0): LoRA on version 1.0.0, merged. New sources: HoVer, CLadder, MultiRC, CRUXEval, CHORD, an appliance-manual set, ACOS (two variants), a phishing set and ToolRet, plus a replay of the second-stage data
- Third stage, versions 2.1.0 to 2.3.0: a separate LoRA on version 1.0.0, merged, with a readout head over the option-letter tokens

### Benchmark contamination statement
- The second and third stages (all versions) trained on **train splits** (or, where a benchmark has no usable train split, on pools disjoint from the evaluation items) of the datasets behind the Decision Index benchmarks. The first stage already included the BANKING77, CLINC150 and GSM8K train splits. Scores on these benchmarks therefore reflect in-distribution training and are not zero-shot.
- For CLadder, training rows came from the part of the dataset that the Decision Index evaluation does not select.
- The licenses of the individual training datasets apply to their own data; the license below covers these model weights.

## Evaluation

Self-scored on 2026-10-12 with the official Decision Index kit (edition 0.3), bf16, one forward pass per question, no temperature scaling. All 140,620 requests were answered with no errors. This is not an official leaderboard submission. The submitted run applied the 2.3.0 LoRA adapter to the 1.0.0 weights at load time. The merged weights in this repository gave the same top choice with a maximum logit difference of 0.10 on the 3 inputs we checked; the full suite was not re-run on the merged files.

| | CoCo-Decision-4B 2.3.0 | CoCo-Decision-4B 2.2.0 | CoCo-Decision-4B 2.1.0 | CoCo-Decision-4B 2.0.0 | CoCo-Decision-4B 1.0.0 |
|---|---:|---:|---:|---:|---:|
| Public index (balanced skill) | **55.38** | 55.05 | 53.93 | 51.93 | 47.8 |
| Raw index | 66.43 | 66.13 | 65.19 | 63.87 | 60.78 |
| Knowledge & reasoning | 34.8 | 34.4 | 33.3 | 31.1 | 28.6 |
| Language understanding | 63.2 | 63.0 | 63.2 | 61.5 | 58.5 |
| Retrieval & classification | 59.9 | 59.0 | 57.1 | 58.4 | 51.4 |
| Tools & automation | 76.8 | 77.5 | 75.0 | 71.1 | 64.5 |
| Arts & taste | 40.2 | 39.0 | 38.8 | 32.9 | 32.0 |

- Latency (RTX 5090, sequential requests): median 25.6 ms, mean 93.2 ms, p95 299.6 ms.
- These are self-reported public-panel figures. The leaderboard's main ranking uses the maintainers' own runs, including a private panel.
- The 1.0.0 column was scored on the 140,178-request kit of 2026-10-08; 2.0.0 to 2.3.0 were scored on 140,620 requests.

## Versions

| Version | Date | Notes |
|---|---|---|
| 2.3.0 | 2026-10-12 | Retrained third LoRA stage, same structure as 2.2.0. Merged bf16 weights, text-only. Public index 55.38 (self-scored) |
| 2.2.0 | 2026-10-11 | Retrained third LoRA stage, same structure as 2.1.0. Merged bf16 weights, text-only. Public index 55.05 (self-scored) |
| 2.1.0 | 2026-10-10 | Different third LoRA stage on version 1.0.0; readout head and bidirectional full attention. Merged bf16 weights, text-only. Public index 53.93 (self-scored) |
| 2.0.0 | 2026-10-09 | Third LoRA stage on additional Decision Index source train splits. Merged bf16 weights, text-only. Public index 51.93 (self-scored) |
| 1.0.0 | 2026-10-08 | First release. Merged bf16 weights, text-only |

## License

CC BY-NC 4.0

## Citation

```bibtex
@misc{cocodecision4b2026,
  title   = {CoCo-Decision-4B: a decision model tuned for the Jev Decision Index},
  version = {2.3.0},
  author  = {{Corners AI}},
  year    = {2026},
  url     = {https://huggingface.co/corners-ai/CoCo-Decision-4B}
}
```
