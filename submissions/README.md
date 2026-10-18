# Model submissions

## Wyat-1-800M — public edition 0.3

| Item | Submission |
| --- | --- |
| Model and bundled native runtime | [teilomillet/wyat-1-800m at `9b112c58`](https://huggingface.co/teilomillet/wyat-1-800m/tree/9b112c5850eef2b47e4e456a2a16151f83d16209), public weights; trained components have the noncommercial terms in the model card and NOTICE |
| Complete public results | [Run directory at `a0b54dfb`](https://huggingface.co/datasets/teilomillet/wyat-1-800m-decision-index-0.3/tree/a0b54dfbd3a899af9af6c1bfc97cd3655151ce16/runs/wyat-1-800m-public03), including [scores.json](https://huggingface.co/datasets/teilomillet/wyat-1-800m-decision-index-0.3/blob/a0b54dfbd3a899af9af6c1bfc97cd3655151ce16/runs/wyat-1-800m-public03/scores.json) with `"complete": true` |
| Inference code | [Native engine](https://github.com/teilomillet/decision-index/blob/2be80b0977bd42921b60932484138b2f58e930ce/examples/wyat_engine.py) and [artifact preparation](https://github.com/teilomillet/decision-index/blob/2be80b0977bd42921b60932484138b2f58e930ce/examples/prepare_wyat.py), commit `2be80b0977bd42921b60932484138b2f58e930ce`; official kit base `f5a6e4fd85fed05e77878dfc4966d0568bfb45ec` |
| Hardware | One NVIDIA GeForce RTX 4070 Ti, 12,282 MiB, single process |
| Public result | Public Decision Index **25.78**, raw index **44.22**; 140,178 scoreable requests, all successful, coverage 100%; 140,620 raw requests, all successful, before 442 official scoring exclusions |

All 43 displayed benchmark populations completed; 37 contribute to the headline
public index. The unmodified compact result log is supplied as `results.jsonl.gz`;
its decompressed SHA-256 is
`5e28a651d9835bd28119bcb7a0bc57980b084ca3cf0d28871299a64f2733d418`.
There were no errors, unsupported requests, abstentions or retries. The rebuilt
suite and benchmark input payloads are not distributed.

### Settings and declared limits

The engine calls the bundled native loader and decision head directly, preserving
every supplied question ID and option. Choice returns all native probabilities
and its selected key; Noul returns native `P(true)` without threshold remapping.
Settings: one request at a time, temperature 1.0, SDPA, uncompiled, one pass, no
permutations or added calibration; native CUDA BF16 backbone linears with FP32
adapters and head. `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` and
`TOKENIZERS_PARALLELISM=false`. The run used reference causal-convolution and
gated-delta kernels, without optional `causal_conv1d` or
`flash-linear-attention` packages.

Measured environment: Python 3.12.13, Torch 2.12.1+cu130, CUDA 13.0,
Transformers 5.18.0, Tokenizers 0.23.2, Safetensors 0.8.0 and Hugging Face Hub
1.33.0. The released bundle's CPU requirements pin Torch 2.14.1; this evaluation
used the CUDA environment above instead. Exact model, runtime and capacity
metadata are in the uploaded `environment.json`.

The original artifact enables `api-bank-schema-next-v1`. To satisfy the kit's
uniform-input rule, `prepare_wyat.py` disables that renderer globally in a
separate manifest, preserving every weight and tokenizer digest and leaving the
source release unchanged. No benchmark-specific prompts, truncation or option
filtering are applied. The adapter rejects an active renderer.

- Source identity: `fork-197dbfe2735120311093338f20d4f2918755bb488d88d450c710f19fc9ee8ff6`.
- Evaluated identity: `fork-4ae9f9a93c1b538356bf44998212f259498fc299ec78cbfd3322d6d7b8af341b`.
- Evaluated manifest SHA-256: `72258ee5398b72d4c626b95a9a2bb0a9769ba19dcd0fee3a1b14f187a1a70bcd`.
- Choice: 2–255 options. Noul: exactly `false` and `true`.
- Complete native prefix plus each branch: at most 32,768 tokens.
- Native branch batch: at most 32 branches and 32,768 padded tokens.
- Each text segment: at most 1,048,576 UTF-8 bytes. No additional question-count cap.

Declared capacity refusals are unsupported; malformed inputs and runtime failures
remain errors. Neither occurred in the public run.

### Reproduction

Use the measured CUDA dependencies above and obtain the private rebuilt edition
0.3 suite using the [kit instructions](../README.md#quickstart-local). The
[engine notes](../examples/wyat.md) describe artifact preparation in more detail.

```sh
git clone https://github.com/teilomillet/decision-index.git
cd decision-index
git checkout 2be80b0977bd42921b60932484138b2f58e930ce
python -m pip install -e .
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="teilomillet/wyat-1-800m",
    revision="9b112c5850eef2b47e4e456a2a16151f83d16209",
    local_dir="/path/to/wyat-release",
)
PY
export PYTHONPATH=/path/to/wyat-release/runtime:.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python examples/prepare_wyat.py --source /path/to/wyat-release/model \
  --out /path/to/wyat-release/model-verbatim
python -u -m decision_index pipeline --edition 0.3 \
  --engine examples.wyat_engine:WyatEngine \
  --model /path/to/wyat-release/model-verbatim --option device=cuda \
  --suite-dir /path/to/private/suite-0.3 \
  --compact --out runs/wyat-1-800m-public03
```

Public-run latency on the RTX 4070 Ti was median **57.8 ms**, mean **249.9 ms**
and p95 **823.7 ms** per request, including tokenization, every question/option
and CUDA synchronization, excluding model loading. These are public-run
measurements. Private scores and official latency eligibility (median, mean and
p80 each at most 1,000 ms on the maintainers' RTX PRO 6000) await maintainer
evaluation.
