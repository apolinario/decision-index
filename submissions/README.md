# Model submissions

## Wyat-2-1.4B — public edition 0.3

| Item | Submission |
| --- | --- |
| Model and bundled native runtime | [wyat-io/wyat-2-1.4b at `5cf21032`](https://huggingface.co/wyat-io/wyat-2-1.4b/tree/5cf21032a3edb7e3dcc47564c8f0e248c46705bf), public weights, Apache-2.0 |
| Complete public results | [Run directory at `e42f2265`](https://huggingface.co/datasets/wyat-io/wyat-2-1.4b-decision-index-0.3/tree/e42f2265c3f69bca31346770a0ddcdf3b2966491/runs/wyat-2-1.4b-public03), including [scores.json](https://huggingface.co/datasets/wyat-io/wyat-2-1.4b-decision-index-0.3/blob/e42f2265c3f69bca31346770a0ddcdf3b2966491/runs/wyat-2-1.4b-public03/scores.json) with `"complete": true` |
| Inference code | [Native engine](https://github.com/teilomillet/decision-index/blob/51166e4e9c95d7dfbac01f4827b8a7463d6641f3/examples/wyat2_engine.py) and [notes](https://github.com/teilomillet/decision-index/blob/51166e4e9c95d7dfbac01f4827b8a7463d6641f3/examples/wyat2.md), commit `51166e4e9c95d7dfbac01f4827b8a7463d6641f3`; official kit base `25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d` |
| Hardware | One NVIDIA GeForce RTX 4070 Ti, 12,282 MiB, single process |
| Public result | Public Decision Index **27.58**, raw index **46.11**; 140,178 scoreable requests, all successful, coverage 100%; 140,620 raw requests: 140,603 successful and 17 declared-capacity refusals, all 17 within the 442 official scoring exclusions |

All 43 displayed benchmark populations completed; 37 contribute to the headline
public index. The unmodified compact result log is supplied as `results.jsonl.gz`;
its decompressed SHA-256 is
`bcd2599c6ae28b091907c32f24a9857ca59603195a3527c8176ec03e93daf19c`.
There were no errors or abstentions. The run was resumed once with the kit's
own resume: 1,817 rows come from an earlier attempt with the same engine and
bundle (every row records both), and every run ID appears exactly once. It was scored with the
official kit's `score` command; a fresh rescore reproduces `scores.json`,
`index.json` and `benchmark-summary.json` exactly. The rebuilt suite and
benchmark input payloads are not distributed.

### Engine used for the run, and the submitted inference code

The public run's rows record `keme.experiments.ouro.jdi_engine:OuroEngine`,
the training project's engine for this model (bundle SHA-256
`5bc679c29cd762f153a8a34c6be6e13f152d918e476876e57662403c5e8d5f0f`), run
through the official kit runner (`runner.py`
`f4c9f36c4c557f537f1d8184db7bf00c5a84d9f4f0aa67f8c06198cead25818d`, files
identical to `25c6ec7`). That engine is not public. The submitted inference
code is the native adapter above with the runtime bundled in the model
repository, and it was qualified against the engine before submission:

- 420 requests (300 rows of the training project's gate set and 120 long
  public requests, covering single-pass, chunked and host-cache context paths):
  scoring logits bitwise identical, argmax 420/420, maximum absolute difference
  0.0;
- 86 official public-suite requests across all 43 benchmarks, run through
  `examples.wyat2_engine:Wyat2Engine` with the packaged runtime: answers and
  choices 86/86 identical to this run, maximum probability difference 0.0.

Those checks ran wyat runtime `d2be944d`. The bundle in the model repository is
`6767e999`, whose only change is the default macOS memory-admission ceiling
(0.85 to 0.90); that guard is inactive on Linux and no scoring code changed.

### Settings and declared limits

The adapter calls the bundled native loader and decision head directly,
preserving every supplied question ID and option. Choice returns all native
probabilities and its selected key; Noul returns native `P(true)` without
threshold remapping. Settings: one request at a time, temperature 1.0, one pass,
no permutations, no added calibration, uncompiled. Each question's context is
prefilled once and its candidates are scored together against it. CUDA BF16
backbone under autocast with FP32 adapters and head; TF32 disabled and
deterministic algorithms (`deterministic-no-tf32-v1`); contexts up to 1,024
tokens are prefilled in one pass, longer ones in 512-token chunks with Flash
attention, and contexts above 12,288 tokens keep their key-value cache in host
memory. These execution settings are declared in the artifact manifest
(`model_id` `ouro-23d704b35b4d3181567d92039e8d60414248a31c215985a80b29c44681cff27c`)
and applied by the loader. `OMP_NUM_THREADS=1`.

Measured environment: Python 3.12.13, Torch 2.8.0+cu128, CUDA 12.8,
Transformers 4.57.6, the versions the bundled runtime pins
(`runtime/requirements.txt`). Exact model and capacity metadata are in the
uploaded `environment.json`.

The model has no task-specific input renderer. No benchmark-specific prompts,
truncation or option filtering are applied.

- Choice: 2–255 options. Noul: exactly `false` and `true`.
- Context plus each candidate: at most 32,768 tokens, without truncation.

Declared capacity refusals are unsupported; malformed inputs and runtime
failures remain errors. The public run had 17 capacity refusals (complete
candidate sequences of 35,845–36,844 tokens), all of them officially excluded
requests, and no errors.

### Reproduction

Use the environment above and obtain the private rebuilt edition 0.3 suite
using the [kit instructions](../README.md#quickstart-local). The
[engine notes](../examples/wyat2.md) describe the adapter in more detail.

```sh
git clone https://github.com/teilomillet/decision-index.git
cd decision-index
git checkout 51166e4e9c95d7dfbac01f4827b8a7463d6641f3
python -m pip install -e .
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="wyat-io/wyat-2-1.4b",
    revision="5cf21032a3edb7e3dcc47564c8f0e248c46705bf",
    local_dir="/path/to/wyat-2-1.4b",
)
PY
python -m pip install -r /path/to/wyat-2-1.4b/runtime/requirements.txt
export PYTHONPATH=/path/to/wyat-2-1.4b/runtime:.
export OMP_NUM_THREADS=1
python -u -m decision_index pipeline --edition 0.3 \
  --engine examples.wyat2_engine:Wyat2Engine \
  --model /path/to/wyat-2-1.4b/model --option device=cuda \
  --suite-dir /path/to/private/suite-0.3 \
  --compact --out runs/wyat-2-1.4b
```

On CUDA, leave `PYTORCH_CUDA_ALLOC_CONF` without `expandable_segments`: the
runtime then replays repeated short candidate batches as CUDA graphs, verified
bitwise against eager scoring before use. With `expandable_segments` it falls
back to eager scoring with identical answers.

Public-run latency on the RTX 4070 Ti was median **136.7 ms**, mean
**577.3 ms**, p80 **651.8 ms** and p95 **1,579.8 ms** per request (successful
requests, device-synchronized in-process wall time including tokenization and
every question and option, excluding model loading). These are public-run
measurements on the training project's engine. Private scores and official
latency eligibility (median, mean and p80 each at most 1,000 ms on the
maintainers' RTX PRO 6000) await maintainer evaluation.
