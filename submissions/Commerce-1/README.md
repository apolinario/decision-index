# InferCrane Commerce-1

[Commerce-1](https://huggingface.co/infercrane/Commerce-1/tree/47e8004868765fbd209ad5421a5af3e27f872e1e)
is a 27B BF16 decision model. It uses a noncausal full-attention backbone and
scores the supplied options directly with a calibrated decision readout. It
does not generate explanations or use a thinking pass.

| | |
|---|---|
| Decision Index 0.3 public index | **61.69** (raw 71.28) |
| Requests | complete: 140,178 of 140,178 scoreable requests `ok`; 0 unsupported; 0 errors; nothing truncated |
| Kit | [`apolinario/decision-index@9eb2dbe`](https://github.com/apolinario/decision-index/commit/9eb2dbe2a358004c8782c66e40a83ac07b953fec) |
| Evaluation engine | `training.commerce_v3_clean.evidence_sharded_modal:CommerceV3ShardedEngine`; `evidence_sharded_modal.py` SHA-256 `46b1918a1f72d6d252c8517eebde2b9e2a8615b8b038252f77ef4b96ccb6ebcf` |
| Public runtime | [`infercrane/commerce-1@5c4a24c19b0d30dfbf00f6d75e25aaf29ce421db`](https://github.com/infercrane/commerce-1/commit/5c4a24c19b0d30dfbf00f6d75e25aaf29ce421db), `/v1/systemone`, immutable model revision `47e8004868765fbd209ad5421a5af3e27f872e1e` |
| Public-run hardware | 40 NVIDIA H200 shard calls for the reused 0.2.1 source run, plus one NVIDIA H200 worker for the rebuilt-GSM8K continuation; one model replica per worker, row chunks of 64, question microbatch 1 |
| Recorded timing | median 117.4 ms, mean 319.3 ms; synchronized row-chunk wall time amortized across rows, not serial request latency or the maintainers' official measurement |

This is a self-hosted public-suite result, not an official Full score or rank.
The Decision Index maintainers run the private components and the official
single-process RTX PRO 6000 latency and held-out-answer checks.

## Results

The immutable public result is
[`infercrane/commerce-1-decision-index/runs/commerce-1`](https://huggingface.co/datasets/infercrane/commerce-1-decision-index/tree/8de2672f0b3b24275834b045d9e3a3dcd83282fc/runs/commerce-1).
It contains `results.jsonl.gz`, `benchmark-summary.json`, `index.json`,
`scores.json`, `environment.json`, and `status.json`.

The submitted run resumed the exact complete 0.2.1 result under edition 0.3.
It reused 137,540 unchanged result rows, ran the 2,638 rebuilt GSM8K requests,
and scored the resulting 140,178 rows with the pinned 0.3 kit. The source run
used 40 H200 shard calls; the 0.3 continuation used one H200 worker. Every
worker loaded one model replica, used row chunks of 64 and an adaptive question
microbatch whose maximum successful size was one. The compact published result
rows are untouched; suite payloads and raw model output are not redistributed.

The submitted results were produced by the content-bound sharded evaluator,
not by HTTP transport. The pinned `/v1/systemone` server is the public path the
maintainers can start for private evaluation. It loads the same immutable model
package, AutoJev decision source and package-bound calibration; its public
qualification and parity evidence are included with that package.

Content bindings for the uncompressed public-run files:

| File or receipt | SHA-256 |
|---|---|
| Evaluated artifact | `b7fba0cceb1b57834fecacfbcb5e361d94fa185e877e891f4e43fc25c6458a1d` |
| `autojev/model.py` | `f859dcbdc097893dc70bbcae5216566abf5d748a62e025a0e190bdc8a53f9f34` |
| Calibration | `4164c7a2dee99f02920bde48b3ea989e45ef8abe8de4399c1c67e700dc7edf93` |
| Public-run receipt | `0565cceabe89c62da857157af8d1c1a54ca6ad1c5556713c586cea890955fb33` |
| `results.jsonl` | `924ee63bd803209ab33ef115ebb774657e3b61037dd2c2f4988c276a4ad5306e` |
| `scores.json` | `c1644ba5ee8f8dd215f63330d748a122dda6b61d7258e011f465c3270ecfee26` |
| `benchmark-summary.json` | `a6765519b4235b80b50056e4ad5d40ed59f8f23e17d187427b0c18b39b975f2c` |
| `index.json` | `168946a8ba5baffd6f4cf6baad44dc3e879cc2dcf9b917606708d23f37cb6224` |
| `environment.json` | `ff155d77b71f5ee9665b0142137f86d3c31878685dd673fd132b1c40ff445aa6` |
| `status.json` | `a9146411d80d2d1fc22bde5fb5dc1e7d3fa21c01bcc34a5f8001170384422e30` |

## Reproduce

Check out the exact public source and model revisions:

```sh
git clone https://github.com/infercrane/commerce-1.git
cd commerce-1
git checkout --detach 5c4a24c19b0d30dfbf00f6d75e25aaf29ce421db
test "$(git rev-parse HEAD)" = "5c4a24c19b0d30dfbf00f6d75e25aaf29ce421db"

uv sync --frozen --python 3.12 --extra v3-runtime
uv run --frozen --python 3.12 --extra v3-runtime \
  hf download infercrane/Commerce-1 \
  --revision 47e8004868765fbd209ad5421a5af3e27f872e1e \
  --local-dir ../commerce-1-model

export COMMERCE_ONE_MODEL_DIR="$(cd ../commerce-1-model && pwd)"
uv run --frozen --python 3.12 --extra v3-runtime \
  commerce-1-v3-serve \
  --model-dir "$COMMERCE_ONE_MODEL_DIR" \
  --base-dir "$COMMERCE_ONE_MODEL_DIR" \
  --offline \
  --host 127.0.0.1 \
  --port 8000
```

In another shell, check out the exact benchmark kit and run the verified
private copy of the 0.3 suite:

```sh
git clone https://github.com/apolinario/decision-index.git
cd decision-index
git checkout --detach 9eb2dbe2a358004c8782c66e40a83ac07b953fec
python -m pip install -e .

python -m decision_index suite verify \
  --edition 0.3 \
  --dir "$SUITE_DIR"

python -m decision_index pipeline \
  --edition 0.3 \
  --engine http \
  --suite-dir "$SUITE_DIR" \
  --option base_url=http://127.0.0.1:8000 \
  --option model=infercrane/commerce-1 \
  --out runs/commerce-1 \
  --compact
```

Independently re-score the published result:

```sh
hf download infercrane/commerce-1-decision-index \
  --repo-type dataset \
  --revision 8de2672f0b3b24275834b045d9e3a3dcd83282fc \
  --include 'runs/commerce-1/*' \
  --local-dir ../commerce-1-decision-index

python -m decision_index score \
  --edition 0.3 \
  --suite-dir "$SUITE_DIR" \
  --results ../commerce-1-decision-index/runs/commerce-1/results.jsonl.gz \
  --engine http \
  --out rescore
```

## Inference and capacity

- The source and model are immutable, content-verified revisions. The runtime
  loads the verified local package in offline mode with no network fallback.
- The BF16 model uses its package-bound calibration and a one-question
  microbatch. It scores the supplied candidates directly and produces zero
  generated tokens.
- The runtime accepts at most 65,536 input tokens, 64 questions per request,
  255 options per Choice question, 10 levels per Score question, and 1,048,576
  request bytes. It refuses requests outside those bounds instead of
  truncating them. No public 0.3 request reached a declared limit.
- Exact model and training-data lineage, third-party notices, and license
  declarations are included in the pinned model repository. They are
  disclosures, not an official benchmark claim.
