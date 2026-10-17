# Granite Decision LoRA · Granite-4.1-30B

IBM's open [Decision LoRA for Granite-4.1-30B](https://huggingface.co/ibm-granite/granitelib-core-r1.0/tree/main/decision/granite-4.1-30b/lora) ([`ibm-granite/granitelib-core-r1.0`](https://huggingface.co/ibm-granite/granitelib-core-r1.0), adapter `decision/granite-4.1-30b/lora`; Apache-2.0; text), a standard PEFT LoRA (r=64, α=128) on frozen [`ibm-granite/granite-4.1-30b`](https://huggingface.co/ibm-granite/granite-4.1-30b), never merged.

**Decision Index 0.3 (public) = 54.04** (raw 64.52, breadth 52.04). 150,317 scoreable requests, all answered (0 unsupported, 0 errors), `complete: true`; adjusted for chance and coverage. Scored at edition 0.3.2, whose public panel is byte-identical to v0.3.

Per area (skill ×100): knowledge 36.9, language 61.5, retrieval 63.9, tools 71.4, arts 27.9.

> This is the **public** component only. The official 0.3.2 headline is `0.20 × public + 0.80 × private`; the private tests are not released here and are not reflected in this score.

## Pinned artifacts

- **Weights (base):** [`ibm-granite/granite-4.1-30b`](https://huggingface.co/ibm-granite/granite-4.1-30b) — public, Apache-2.0.
- **Adapter:** [`ibm-granite/granitelib-core-r1.0`](https://huggingface.co/ibm-granite/granitelib-core-r1.0), path `decision/granite-4.1-30b/lora` — public, Apache-2.0. Loaded unmerged.
- **Results (`scores.json`):** [eval-scripts-granite-4-1 branch](https://github.com/subhajitchaudhury/decision-index/blob/eval-scripts-granite-4-1/submissions/eval-scripts-granite-4-1/scores.json) of this fork.
- **Inference code:** [`decider_granite_decision_lora.py`](https://github.com/subhajitchaudhury/decision-index/blob/eval-scripts-granite-4-1/submissions/eval-scripts-granite-4-1/decider_granite_decision_lora.py) and the end-to-end [`eval_granite_decision_lora.sh`](https://github.com/subhajitchaudhury/decision-index/blob/eval-scripts-granite-4-1/submissions/eval-scripts-granite-4-1/eval_granite_decision_lora.sh), on the same branch.
- **Kit:** [`apolinario/decision-index`](https://github.com/apolinario/decision-index), used as an importable module (no fork, no patches).
- **Suite:** edition `release-v3.2` (0.3.2 public panel), `rows_sha256` `b2b56d6f…0792d5`, `added_sha256` `7429f3c9…f4ce89` as recorded in `scores.json`.
- **Engine id:** `granite41-30b-decision-lora-granitedec`.

## Inference configuration

- **Hardware:** 1× NVIDIA H100 80GB, tensor-parallel size 1, one engine.
- **Serving:** stock vLLM 0.30 serving base + adapter via `--enable-lora --max-lora-rank 64 --max-loras 1 --lora-modules decision-lora=<adapter>` (not merged, not the aLoRA fork), `--max-model-len 32768`, `--gpu-memory-utilization 0.90`.
- **Prompt:** card-exact — one user turn, no system message, lettered options `A: desc`, `add_generation_prompt=True`.
- **Decoding:** single-token option-label logit readout, one forward pass per question (`max_tokens=1`, `temperature=0`), softmax over the single-token letter labels. The model generates nothing — it reads the state once and scores every supplied option.
- **Counts:** 150,317 scoreable, 150,317 ok, 0 unsupported, 0 errors, 0 abstained. Nothing truncated at 32,768 tokens.

## Reproduction

The adapter and base are public, so anyone can reproduce the full run end-to-end on one 80 GB GPU. Prerequisites: a Python env with `vllm>=0.30`, `transformers>=4.44`, `huggingface_hub`, `httpx`, `numpy`, `git`, and an HF login (`huggingface-cli login` or `HF_TOKEN`) with access to the gated suite dataset.

```sh
# fetch the eval script + decider from the eval-scripts-granite-4-1 branch of this fork, then:
bash eval_granite_decision_lora.sh
```

The script clones the kit, downloads and verifies the frozen suite, downloads the adapter, serves base + adapter on stock vLLM, runs the decider, and scores at edition 0.3.2 — printing the Decision Index and per-area skills. See the script header for env overrides (`BASE`, `SUITE`, `KIT`, `WORK`, `ADAPTER`, `PORT`, `CONC`, `MAXLEN`, `EDITION`).

**Re-score the posted results** (no model access needed — this is what reviewers run against the uploaded `results.jsonl`):

```sh
python -m decision_index score --suite-dir <suite> --edition 0.3.2 \
    --results runs/granite41-30b-decision-lora/results.jsonl --out runs/granite41-30b-decision-lora
```

## Integrity and limitations

- **Complete, untouched run:** `complete: true`, every one of the 150,317 scoreable requests answered (0 unsupported, 0 errors). The suite hashes above are recorded in `scores.json` and are verified by the kit on download.
- **No merging, no custom decoding:** the adapter is loaded unmerged on stock vLLM; scoring is a single forward pass with no chain-of-thought, tools, or generation. No kit patches.
- **Weakest areas:** arts (skill 27.9) and knowledge (36.9) — e.g. POP909-CL, Humicroedit, HLE, ChessBench score at or near chance. These drag the chance-and-coverage-adjusted index well below the raw 64.52.
- **Scope:** public component only. The 80% private portion of the official 0.3.2 headline is not computed here; the base weights and adapter are public, so maintainers can run the private half themselves.
