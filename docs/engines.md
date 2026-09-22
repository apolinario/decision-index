# Engines

An engine is a callable `engine(state, questions) -> (response, raw)`; see `decision_index/engines/base.py`. The runner validates every response (`validate`): every question answered, every option carrying a finite probability, probabilities summing to 1 within 0.01, chosen key among the options.

Exceptions have meaning: `Unsupported` marks a declared capacity limit (context window, option count, question count) and becomes status `unsupported`; `NativeAbstention` marks a model that chose an abstention outcome outside the supplied options (`abstained`); anything else is an `error`. Errors are retried on resume, the others are final. Five errors before the first success stop the run; an out-of-memory or device assert stops it too, so the accelerator context is not reused after corruption.

## `http`

`POST {base_url}/v1/systemone` with `{"model", "state", "questions"}` and optional extra fields (`--option extra='{"samples":1}'`), bearer token from `DECISION_INDEX_API_KEY`. HTTP 400/413/422 whose body mentions a known capacity marker (`maximum context length`, `options per choice`, `context window`, ...) is `Unsupported`; other non-2xx responses raise. The response body is used as-is (minus `evaluation_trace`), so the server owns the answer format.

```sh
python -m decision_index run --engine http --option base_url=http://127.0.0.1:8000 --option model=my-model
```

## `transformers`

Parallel constrained decoding on a stock causal LM (`AutoModelForCausalLM`). For each question:

1. Render one fixed prompt: a system line, then `State:`, the state (JSON if not a string), `Question:`, the instructions, `Options:` as `key: description` lines, and `Reply with exactly one option key.`; wrapped in the tokenizer's chat template when it has one, otherwise plain text with an `Answer:` suffix.
2. Tokenize the prompt; tokenize every option key. If `len(prompt) + max(len(option)) > limit`, raise `Unsupported`. The limit is `max_position_embeddings` or `--option max_tokens=N`, whichever is smaller.
3. Run **one forward pass** over `[prompt tokens] + [option 1 tokens] + [option 2 tokens] + ...` with a 4D boolean attention mask: prompt tokens are causal, each option's tokens attend to the prompt and causally to their own option only, and position ids restart at the prompt length for every option. `logits_to_keep` selects only the positions that predict option tokens.
4. Score each option by the summed log-probability of its key tokens; probabilities are the softmax over options; the choice is the argmax.

The state prefix is cached across the questions of one request (KV cache cropped to the longest common token prefix of consecutive prompts), which is exact and cuts the cost of multi-question rows (ContractNLI, ACOS, SATA, BFCL, ToolRet chunks). On CUDA out-of-memory the option block is re-run in halves down to one option at a time, then refused as `Unsupported`.

Verified: the one-pass scores equal naive per-option scoring within 1e-5 (float32) on Qwen2.5-0.5B-Instruct, and CPU and MPS agree within 1e-4.

Options: `--model`, `--option revision=...`, `--option device=cuda|mps|cpu`, `--option dtype=bfloat16|float16|float32` (defaults: bfloat16 on CUDA, float32 elsewhere), `--option attn=sdpa|eager`, `--option max_tokens=N`, `--option cache_prefix=false`, `--option chat_template=false`.

## `random`

Uniform choice over the options with a fixed seed. A sanity check for the pipeline, not a model.

## Writing your own

```python
from decision_index.engines import Engine, Unsupported

class MyEngine(Engine):
    name = "mine"
    latency = "what the timing covers"

    def __init__(self, **options):
        super().__init__(**options)
        self.provenance = {"repo": "...", "revision": "..."}

    def __call__(self, state, questions):
        answers = {}
        for key, q in questions.items():
            if len(q["criteria"]) > 100:
                raise Unsupported("declared limit of 100 options")
            probs = ...
            answers[key] = {"type": "choice", "choice": max(probs, key=probs.get), "probabilities": probs}
        return {"model": "mine", "answers": answers}, None
```

Run it with `--engine my_package.my_module:MyEngine`. Do not truncate, do not drop options, do not adapt the prompt per benchmark; refuse instead.
