# Wyat-2 engine

`wyat2_engine.py` runs a `wyat-ouro-decision-v2` artifact, such as
wyat-2-1.4b (Ouro-1.4B with step adapters and a scalar decision head), through
wyat's native loader. It does not generate answers with a language-model
prompt. It passes the complete `state` and `questions` directly to
`model.predict`, preserving question IDs and option order. Choice returns every
native probability and its native selected key; Noul returns the native
probability of true.

A v2 manifest declares the context execution the model was trained with, and
that execution is part of the artifact identity. The loader applies it as
declared:

- `execution.context_prefill`: a context of at most
  `single_chunk_max_tokens` is prefilled in one pass, a longer one in chunks of
  `chunk` tokens with the declared chunked attention, and a context above
  `host_cache_above_tokens` keeps its key-value cache in host memory between
  layers;
- `execution.numerics`: `deterministic-no-tf32-v1` disables TF32 and uses the
  exact SDPA math kernel.

The adapter refuses a v1 artifact, an artifact with a task-specific input
renderer (the kit forbids per-benchmark prompt adaptations), and devices other
than `cpu` and `cuda`. It records the manifest hash, the artifact's execution,
the hash of every runtime source file and the settings below in the run's
`environment.json`.

From the kit repository root, with wyat on `PYTHONPATH` and the artifact
directory at `/path/to/wyat-2-1.4b`:

```sh
python -m decision_index pipeline --edition 0.3 \
  --engine examples.wyat2_engine:Wyat2Engine \
  --model /path/to/wyat-2-1.4b --option device=cuda \
  --suite-dir /path/to/private/suite-0.3 \
  --compact --out runs/wyat-2-1.4b
```

Inference is one request at a time, temperature 1, one pass without
permutations, SDPA, uncompiled, with the artifact's CUDA dtype policy (BF16
backbone under autocast, FP32 adapters and decision head). Each question's
context is prefilled once and its candidates are scored together against it.
No prompt selection, calibration or truncation is added. Every question and
option is scored. Choice accepts 2 through 255 options, and each context plus
candidate must fit the artifact's `max_length` (32,768 tokens for
wyat-2-1.4b). Requests over declared capacity are unsupported; malformed inputs
and device failures remain errors.

On CUDA, the fastest path needs `PYTORCH_CUDA_ALLOC_CONF` without
`expandable_segments`. The loader replays short candidate batches whose shapes
repeat as CUDA graphs, and capture needs the default allocator segments. With
`expandable_segments` set, the loader falls back to eager scoring without a
message: answers and probabilities are identical, but those batches run slower.

The adapter's laws use [tlaw](https://github.com/teilomillet/tlaw) 0.0.12. The
first file checks the adapter alone against a stand-in model; the second loads
a tiny random v2 artifact on CPU and needs a directory holding Ouro's
`configuration_ouro.py` and `modeling_ouro.py`:

```sh
tlaw check tests/wyat2_engine_laws.py
WYAT_OURO_SOURCE=/path/to/ouro-source tlaw check tests/wyat2_runtime_laws.py
```

The rebuilt suite stays private. `--compact` omits input payloads and raw
outputs; review result rows and environment metadata before uploading.
Maintainers determine private scores and the official RTX PRO 6000 latency
eligibility.
