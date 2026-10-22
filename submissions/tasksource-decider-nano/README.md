# tasksource-decider-nano

[tasksource/tasksource-decider-nano](https://huggingface.co/tasksource/tasksource-decider-nano/tree/v5) is a 150M-parameter
joint cross-encoder fine-tuned from `cross-encoder/ettin-reranker-150m-v1`. It reads a request's state once and
scores every option of every question against it in the same forward pass. It does not generate text. This is v5,
trained for 64,000 steps on an updated 480,000-request mix; it replaces v4 (30.90, #120) on the board. The
multi-vector model of #70 and #76 stays at tag `v1` of the same repository
(renamed from `tasksource/tasksource-jev-nano-v1`; the old URL redirects).

| | tasksource-decider-nano |
|---|---:|
| Decision Index 0.3, public index | **33.51** |
| Area skill (0.3) | Knowledge & Reasoning 9.7 · Language Understanding 38.6 · Retrieval & Classification 50.1 · Tools & Automation 50.9 · Arts & Human Taste 16.6 |
| Requests | 140,171 `ok`, 7 `unsupported`, 0 errors |
| Request time, 1× NVIDIA A30 24 GB (shared), one request at a time | median 23 ms · mean 33 ms · p80 37 ms |
| Results | [runs/tasksource-decider-nano](https://huggingface.co/datasets/tasksource/decision-index-results/tree/49d7bcbbf84f2a57b1463f36a724d11c045313f7/runs/tasksource-decider-nano) |

The run used kit `9eb2dbe` (edition 0.3) with the engine shipped in the model repository:

```sh
hf download tasksource/tasksource-decider-nano --revision 0dd9e308c2bcc40cd8404e78ed6dc64828fa326a --local-dir tasksource-decider-nano
cd tasksource-decider-nano
python -m decision_index pipeline --engine decision_index_engine:DeciderEngine --option repo=. --edition 0.3 --out runs/tasksource-decider-nano
```

Settings: bf16 autocast, 8,192-token context, **no truncation**. A request with a (state, question, option) pair
above 8,192 tokens raises `Unsupported`; that is the only source of the 7 unsupported requests.

Latency was measured on a shared A30, which is slower than the board's RTX PRO 6000. The engine encodes the state
once per request, so long states with many options (API-Bank, BRIGHT) stay fast.

Training data: tasksource-jev-typed-decisions, the *train* splits of public datasets behind the index benchmarks,
tool-use and LLM-routing train sets, tasksource-synthetic-typed-decisions, and 0.5% LocalLLaMA/typed-decisions *train*. Nothing comes from evaluation
splits, and benchmarks without a train split stay evaluation only. Every training request passed an exact and
6-gram content firewall against the 0.3 suite. Details are in the model card.
