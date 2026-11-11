# Ainvest-Decision-31b-v2.0

Text-only Decision Index submission. Please evaluate this model on the maintainers’ private same-skills and new-domain tests for the Full score.

- [Model and model card](https://huggingface.co/Ainvest-World/Ainvest-Decision-31b-v2.0)
- [Pinned weights and inference package](https://huggingface.co/Ainvest-World/Ainvest-Decision-31b-v2.0/tree/8e1ed5b2a6a268282010635206f2362cf88e5d12) at `8e1ed5b2a6a268282010635206f2362cf88e5d12`
- [Complete public scores](https://huggingface.co/datasets/Ainvest-World/Ainvest-Decision-31b-v2.0-decision-index-results/blob/4ad6c627ac1e78f3a1c9de44d18a955db6882980/runs/gemma-b4857/scores.json): Public **71.58**, raw **78.19**, `complete: true`
- [Original complete run](https://huggingface.co/datasets/Ainvest-World/Ainvest-Decision-31b-v2.0-decision-index-results/tree/4ad6c627ac1e78f3a1c9de44d18a955db6882980/runs/gemma-b4857): 140,178 requests / 312,320 subquestions, all successful
- [Raw result archive](https://huggingface.co/datasets/Ainvest-World/Ainvest-Decision-31b-v2.0-decision-index-results/resolve/4ad6c627ac1e78f3a1c9de44d18a955db6882980/runs/gemma-b4857/results.jsonl.gz)
- [Publication manifest](https://huggingface.co/datasets/Ainvest-World/Ainvest-Decision-31b-v2.0-decision-index-results/blob/4ad6c627ac1e78f3a1c9de44d18a955db6882980/publication.json) and [dedicated-H200 validation](https://huggingface.co/datasets/Ainvest-World/Ainvest-Decision-31b-v2.0-decision-index-results/blob/4ad6c627ac1e78f3a1c9de44d18a955db6882980/runs/gemma-b4857/publication-dedicated-verification-and-timing.json)

## Model and training

Building on our [Ainvest-Decision v1.0](https://huggingface.co/Ainvest-World/Ainvest-Decision-27b-v1.0) text-decision modeling work, v2.0 replaces the Qwen3.8-27B base with Gemma 4 31B IT and enables per-question read twice. The new Gemma base was trained in BF16 with full-parameter fine-tuning of the language backbone; token embeddings and the output head were kept frozen. Training used 2M+ cleaned text-decision examples and teacher-generated soft-label probability distributions. Only text decision capabilities were trained and evaluated.

Native label-token logits after `Answer:` are normalized over all supplied option labels. Causal full/sliding attention and the checkpoint’s native logit soft-cap are preserved. Thinking is disabled, and no answer tokens are generated.

Each question is rendered independently with its complete state, instructions, question and options. If its complete read-once prompt has at most 3,000 tokens, the non-system input is repeated using the KnowLine read-twice layout. Longer prompts are read once. Long-request question headers, option rotations and extra temperature calibration are disabled.

## Installation and inference

Use Python 3.12/Linux/CUDA. Install a CUDA-compatible PyTorch 2.14.0 build first; FlashAttention may require a matching CUDA toolkit. Validated dependencies include Transformers 5.17.0, FlashAttention 2.8.3.post2 and Triton 3.8.0. BF16 checkpoint size is approximately 62.55 GB.

```bash
python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("Ainvest-World/Ainvest-Decision-31b-v2.0", revision="8e1ed5b2a6a268282010635206f2362cf88e5d12"))'
AINVEST_MODEL_DIR=/absolute/path/to/snapshot
pip install --no-build-isolation -r "$AINVEST_MODEL_DIR/requirements.lock.txt"
pip install --no-deps "$AINVEST_MODEL_DIR"
```

The Python API is `ainvest_decision.engine:AinvestDecisionEngine`; `__call__(state, questions)` returns `(response, raw)`. `choice` answers retain the original option keys and a probability for every option. `noul` returns P(true).

## Full public reproduction

Install the official Decision Index kit separately. Build and import the frozen public 0.3 suite following the [upstream instructions](https://github.com/apolinario/decision-index#quickstart-local). With the imported suite in the kit’s default 0.3 directory:

```bash
python -m decision_index pipeline --edition 0.3 \
  --engine ainvest_decision.engine:AinvestDecisionEngine \
  --option model=Ainvest-World/Ainvest-Decision-31b-v2.0 \
  --option revision=8e1ed5b2a6a268282010635206f2362cf88e5d12 \
  --option device=cuda:0 --option dtype=bfloat16 \
  --option max_length=262144 \
  --option read_twice=1 --option twice_max_tokens=3000 \
  --option batch_size=16 --option batch_tokens=16384 \
  --option sort_window=128 \
  --compact --out runs/ainvest-decision-31b-v2.0
```

This command reads the complete imported suite, runs the model and scores the results locally. It does not upload anything. It is provided for reproduction; publication reused the existing full run rather than performing another full-suite evaluation.

## Original run and publication validation

The full run contains 140,178 successful requests / 312,320 subquestions, with Public 71.58 and raw 78.19. Existing native result and score files are preserved. Original hardware was 8 H200 replicas. Per-question read twice was enabled, with a 16,384-token packing budget, at most 16 questions per pack, and a 128-question sorting window. Native attention uses the evaluated FlashAttention/FlexAttention implementation. Historical timings sum amortized packed-batch forward times; they exclude rendering, tokenization and loading.

An independently installed package passed the official runner on 100 fixed requests / 285 subquestions. Rendered token IDs, hashes, option keys and read-twice decisions matched the sealed inputs. Frozen production-code serial replay matched all package probabilities exactly. The independent3915 run on one dedicated H200 matched that accepted serial reference exactly; the official scorer accepted the sample.

One subquestion changes its chosen option relative to the historical packed results: its original top two BF16 probabilities were exactly tied. Reconstructing the original packed batch reproduces the historical probabilities exactly. The maximum probability difference across the sample versus historical packing was 0.03097697; the variation is documented rather than modifying the original results.

## Local single-request latency

100 requests, processed serially on one dedicated NVIDIA H200; all285 subquestions included:

| Metric | Local measurement |
|---|---:|
| Median | 111.85 ms |
| Mean | 547.79 ms |
| p80 | 481.00 ms |
| p95 | 3125.66 ms |
| Peak allocated memory | 66.08 GB |

Timing includes rendering, tokenization, transfers, all question forwards and synchronization, and excludes loading and warmup. This is a local H200 preflight; the maintainers determine official latency on an RTX PRO 6000.

## Capacity and license

Text `choice` (2–255 options) and `noul`, with up to 64 questions per request. The complete rendered input limit is 262,144 tokens per question, including repeated input and template tokens. If any question exceeds declared capacity, the entire request is unsupported. All options are retained and nothing is truncated. Practical memory requirements depend on input length and hardware.

Weights and Ainvest-authored code are Apache-2.0, with the upstream Gemma and KnowLine notices retained. Evaluation input text and gold labels are not redistributed.
