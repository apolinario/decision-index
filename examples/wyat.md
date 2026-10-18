# Wyat native engine

`wyat_engine.py` wraps the native loader and typed decision head; it does not
generate answers with a language-model prompt. It passes the complete `state`
and `questions` directly to `model.predict`, preserving question IDs and option
order. Choice returns every native probability and its native selected key;
Noul returns the native probability of true.

Use Python 3.12 or newer and the public
[selected Wyat-1-800M snapshot](https://huggingface.co/teilomillet/wyat-1-800m/tree/9b112c5850eef2b47e4e456a2a16151f83d16209),
including its `model/` and `runtime/` directories. The native loader verifies
every artifact file. Use the bundled runtime, whose exact source hashes are
recorded in the run's `environment.json`.

Install the kit and runtime dependencies in a CUDA environment. The release's
pinned CPU requirements use Torch 2.14.1; the public evaluation used Python
3.12.13, Torch 2.12.1+cu130, CUDA 13.0, Transformers 5.18.0, Tokenizers 0.23.2,
Safetensors 0.8.0 and Hugging Face Hub 1.33.0. Use a CUDA-enabled Torch build
instead of the CPU pin. This run used the reference causal-convolution and
gated-delta kernels, with no optional `causal_conv1d` or
`flash-linear-attention` package installed.

The selected release contains a task-specific API-Bank renderer. The kit forbids
per-benchmark prompt adaptations, so prepare a **separate inference artifact**
with that renderer disabled for every request. The helper validates the source,
preserves every weight and tokenizer hash, changes only the rendering setting
and provenance in a new manifest, and recomputes the native artifact identity.
It never modifies the selected release. The adapter rejects an artifact with an
active renderer. The submission must disclose the selected source identity and
this uniformly rendered inference identity.

From the kit repository root:

```sh
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="teilomillet/wyat-1-800m",
    revision="9b112c5850eef2b47e4e456a2a16151f83d16209",
    local_dir="/path/to/wyat-release",
)
PY
export PYTHONPATH=/path/to/wyat-release/runtime:.
python examples/prepare_wyat.py --source /path/to/wyat-release/model \
  --out /path/to/wyat-release/model-verbatim
python -m decision_index pipeline --edition 0.3 \
  --engine examples.wyat_engine:WyatEngine \
  --model /path/to/wyat-release/model-verbatim --option device=cuda \
  --suite-dir /path/to/private/suite-0.3 \
  --compact --out runs/wyat-1-800m
```

Inference is one request at a time, temperature 1, one pass without permutations,
SDPA, uncompiled, with the native CUDA dtype policy (BF16 backbone linears,
FP32 adapters and decision head). No prompt selection, calibration or truncation
is added. Every question and option is scored. Native branch batching uses at
most 32 branches and 32,768 padded tokens; each complete prefix plus branch must
fit 32,768 tokens, and each text segment must fit 1,048,576 UTF-8 bytes. Choice
accepts 2 through 255 options. Requests over declared capacity are unsupported;
malformed inputs and device failures remain errors.

The rebuilt suite stays private. `--compact` omits input payloads and raw outputs;
review result rows and environment metadata before uploading. Public results
are only the public component of the index. Maintainers determine private
scores and the official RTX PRO 6000 latency eligibility.
