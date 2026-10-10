# tasksource-decider-nano

[tasksource/tasksource-decider-nano](https://huggingface.co/tasksource/tasksource-decider-nano/tree/v2) is a 150M-parameter
joint cross-encoder fine-tuned from `cross-encoder/ettin-reranker-150m-v1`. It reads a request's state once and
scores every option of every question against it in the same forward pass. It does not generate text. It replaces the
multi-vector model of #70 and #76, which stays at tag `v1` of the same repository
(renamed from `tasksource/tasksource-jev-nano-v1`; the old URL redirects).

| | tasksource-decider-nano |
|---|---:|
| Decision Index 0.3, public index | **26.77** |
| Area skill (0.3) | Knowledge & Reasoning 6.0 · Language Understanding 28.2 · Retrieval & Classification 42.0 · Tools & Automation 44.6 · Arts & Human Taste 13.9 |
| Requests | 140,171 `ok`, 7 `unsupported`, 0 errors |
| Request time, 1× NVIDIA A10 23 GB, one request at a time | median 45 ms · mean 71 ms · p80 71 ms |
| Results | [runs/tasksource-decider-nano](https://huggingface.co/datasets/tasksource/decision-index-results/tree/98b8ddb74c91f2d80e5c1cb6bbccd25aea7130d3/runs/tasksource-decider-nano) |

The run used kit `9eb2dbe` (edition 0.3) with the engine shipped in the model repository:

```sh
hf download tasksource/tasksource-decider-nano --revision afa49af5a6b3550460fafc431216fbf1e8978fdb --local-dir tasksource-decider-nano
cd tasksource-decider-nano
python -m decision_index pipeline --engine decision_index_engine:DeciderEngine --option repo=. --edition 0.3 --out runs/tasksource-decider-nano
```

Settings: bf16 autocast, 8,192-token context, **no truncation**. A request with a (state, question, option) pair
above 8,192 tokens raises `Unsupported`; that is the only source of the 7 unsupported requests.

Latency was measured on a shared A10, which is slower than the board's RTX PRO 6000. The engine encodes the state
once per request, so long states with many options (API-Bank, BRIGHT) stay fast.

Training data: tasksource-jev-typed-decisions, the *train* splits of public datasets behind the index benchmarks,
tool-use and LLM-routing train sets, and tasksource-synthetic-typed-decisions. Nothing comes from evaluation
splits, and benchmarks without a train split stay evaluation only. Every training request passed an exact and
6-gram content firewall against the 0.3 suite. Details are in the model card.
