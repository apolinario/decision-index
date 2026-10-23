# crystal-0.8b

[Kyunghwan-kim/crystal-0.8b](https://huggingface.co/Kyunghwan-kim/crystal-0.8b/tree/befc98ecaf858a5e4cafc11b8921fea709092943) is a 0.8B decision model built on Qwen/Qwen3.5-0.8B.

- **Weights:** a merge of the base model (20%) with two fine-tunes of its whole language model (40% each). Weights are CC BY-NC 4.0; the runtime code is Apache-2.0.
- **Readout:** each question is one prompt. The answer is a softmax over the tied LM-head rows of the question's answer codes (up to 255 single-token codes) at the last prompt token. Nothing is generated.
- **Attention:** the six full-attention layers read the prompt noncausally, and the Gated DeltaNet layers stay causal.
- **Parameters:** 752,393,024 are served (the language model). The checkpoint also keeps the base model's 100,592,896-parameter vision tower, which is never loaded for text, for 852,985,920 in total.

| | crystal-0.8b |
|---|---:|
| Decision Index 0.3, public index | **42.79** (raw 57.26) |
| Area skill (0.3) | Knowledge & Reasoning 19.1 · Language Understanding 44.8 · Retrieval & Classification 55.4 · Tools & Automation 64.9 · Arts & Human Taste 33.4 |
| Requests | 140,178 scoreable, all `ok`, 0 unsupported, 0 errors |
| Request time, 1× NVIDIA RTX A6000, one request at a time | median 31.4 ms · mean 63.4 ms · p80 43.9 ms |
| Results | [runs/crystal-0.8b](https://huggingface.co/datasets/Kyunghwan-kim/crystal-0.8b-decision-index/tree/bb6fdaa908d5adf6b7d0e780c82bdcd7cfcc01e6/runs/crystal-0.8b) (compact: no suite text) |

The run used kit `f5a6e4f` (edition 0.3, unmodified runner and scorer) with the engine shipped in the model repository:

```sh
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install "transformers==5.17.0" safetensors accelerate flash-linear-attention==0.5.2 \
    "decision-index @ git+https://github.com/apolinario/decision-index@f5a6e4f"
hf download Kyunghwan-kim/crystal-0.8b --revision befc98ecaf858a5e4cafc11b8921fea709092943 --local-dir crystal-0.8b
PYTHONPATH=crystal-0.8b python -m decision_index pipeline --edition 0.3 --engine crystal.engine:CrystalEngine \
    --option model=crystal-0.8b --out runs/crystal-0.8b
```

**Settings:**
- BF16 backbone with SDPA attention, FP32 readout and softmax, temperature 1.
- A request's question prompts are batched under a 32,768-token budget, at most 8 per forward pass.
- One fixed prompt for every request: no per-benchmark settings, no option-order averaging, no calibration fit on the suite.

**Declared limits:** text only and at most 255 options per question. Neither limit was reached in the public suite, nothing was truncated and no options were filtered.

**Latency:** measured during the scoring run on an RTX A6000, which is slower than the board's RTX PRO 6000. The slowest requests are long ToolRet and BRIGHT ranking requests (p95 170.6 ms, max 9.4 s).

**Training data:** [TRAINING_DATA.md](https://huggingface.co/Kyunghwan-kim/crystal-0.8b/blob/befc98ecaf858a5e4cafc11b8921fea709092943/TRAINING_DATA.md) lists all 113 public sources with their licences and row counts.
- Every training row is a question built from a dataset's training split (validation where noted).
- The training splits of 18 index benchmarks are among them, so the public index is partly in-distribution:
  - ACOS, API-Bank, Amazon ESCI, BANKING77, CLINC150, ContractNLI, GSM8K, Habermas, HellaSwag, HoVer
  - Humicroedit, New Yorker, NLI4CT, RAGTruth, VAST, When2Call, WinoGrande, iSarcasmEval
- No test split of any index benchmark was used.
- Training rows that overlap the 0.3 public rows were removed: 13-word shingles, plus exact field and line matches for short texts, with digits masked.
- Benchmarks whose items cannot be told apart by text (cfcolor, POP909 and the templated CLadder and BBH items) were excluded from training by name.

**Suite:** rebuilt from the public sources and passes `suite verify`. At `f5a6e4f` the rebuild reproduces the frozen rows only after two fixes:
- RouterBench has to be built under Python 3.12 or later, because its calibration profiles are `sum()` means.
- SGD has to be built with the builder from before `2d7791c`.
- Every request's `payload_sha256` matches the board's published d3-lite run.
