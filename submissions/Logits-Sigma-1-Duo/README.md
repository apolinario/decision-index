# Logits Sigma-1 Duo

[logitshq/Logits-Sigma-1-Duo](https://huggingface.co/logitshq/Logits-Sigma-1-Duo/tree/a0c1b6e8242916aafdcf9a5a49c1a952cb56e091) is an ensemble of
two decision models served together on one GPU (Apache-2.0). It averages their per-option probabilities 50 / 50 and
answers with the argmax; it generates nothing.
- **Member A:** PelaAI/KnowLine-31B-Gen1 @ `8f052a81` with Logits' LoRA (v7 warm) merged at scale 0.5, read with
  Logits' option-code layout.
- **Member B:** vllm-sr/d3-ultra @ `37fee019` with its own runtime.

Both members are FP8 block-128 checkpoints.

| | |
|---|---|
| Decision Index 0.3 public index | **72.93** (raw 79.54). `score --edition 0.3` on the published `results.jsonl.gz` with the kit at `25c6ec7` gives the same |
| Requests | complete: 140,178 of 140,178 scoreable requests answered, 0 unsupported, nothing truncated, 0 errors. 526 requests over 50,000 characters were answered by member A alone |
| Request time, one request at a time | median 293, mean 859, p80 659 ms over the latency-v1 design (750 rows drawn with seed 20260926, the first 10 as warm-up, 740 timed) on one NVIDIA RTX PRO 6000 Blackwell Server Edition, FP8, causal-conv1d 1.7.0; peak GPU memory 87,891 MiB |
| Agreement on those rows | 99.40% of decisions vs the two members in bf16 (FP8 bar 96%); 99.43% (on the 689 rows also in the public run) vs the same engine on an H200 |
| Request time in `scores.json` | median 463.5, mean 876.6 ms: eight engine processes, one per H200, so not an RTX PRO 6000 figure |
| Model | public; every file matches `SHA256SUMS` there; upstream revisions in `UPSTREAMS.json` |
| Results | [`runs/logits-sigma-1-duo`](https://huggingface.co/datasets/logitshq/logits-sigma-1-duo-decision-index/tree/202db5f059665729f8ecca706a4fdb7d0db549e4/runs/logits-sigma-1-duo) @ `202db5f059665729f8ecca706a4fdb7d0db549e4`: `results.jsonl.gz` (compact: no suite text), `scores.json`, `index.json`, `benchmark-summary.json`, `environment.json` (one per shard) |

## Run

With Docker (the serving we measured), from the model repo's `code/`:

```sh
hf download logitshq/Logits-Sigma-1-Duo --revision a0c1b6e8242916aafdcf9a5a49c1a952cb56e091 --local-dir duo
git clone https://github.com/apolinario/decision-index kit && git -C kit checkout 25c6ec7
docker build -f duo/code/Dockerfile --build-context kit=kit -t logits-sigma-1-duo duo/code
docker run --gpus all -v $PWD/duo:/models/duo:ro -v $PWD/runs:/runs logits-sigma-1-duo \
    pipeline --compact --out /runs/logits-sigma-1-duo
```

Without Docker (Python 3.12, NVIDIA driver ≥ 580; causal-conv1d compiles with nvcc 13 and g++):

```sh
uv venv --python 3.12 venv && . venv/bin/activate
uv pip install -r duo/code/requirements.lock ./duo/code
uv pip install --no-deps "decision-index @ git+https://github.com/apolinario/decision-index@25c6ec7"
CAUSAL_CONV1D_FORCE_BUILD=TRUE uv pip install --no-build-isolation causal-conv1d==1.7.0
python -m decision_index pipeline --engine logits_sigma_1_duo.engine_pair:Duo --option model=duo \
    --compact --out runs/logits-sigma-1-duo
```

One 96 GB GPU holds both members (~57 GB of FP8 weights). The run peaks at 87,891 MiB. At load the engine
prints which FP8 matmul and which d3-ultra kernels serve the run, and `environment.json` records them under
`model_source`.

## Engine

`logits_sigma_1_duo.engine_pair:Duo`, shipped in the model repo under `code/` (Apache-2.0).

- **Member A** (`logits_sigma_1_duo.engine:LogitsEngine`, transformers, FP8 weights, bf16 activations): Gemma 4's
  chat template with thinking off. Each question is one prompt, with the options labelled by one-token codes; the
  engine reads the codes' logits at the answer position and softmaxes them over the question's codes. A `noul`
  question is a two-option choice reported as p(yes). A request's questions share the state block, which is read
  once per request (packs ≤ 8,192 tokens, forwards ≤ 16,384).
- **Member B** (`d3_engine:D3Engine`, vllm-sr's runtime from the d3-ultra directory, unchanged): prefix-bidirectional
  attention over the state, code readout, float32 softmax.
- **Mixing:** for choice questions, the mean of the two members' option probabilities, argmax; for `noul`, the mean of
  p(yes). Weights are fixed at 0.5 / 0.5 for every benchmark.
- **One size rule:** a request whose compact JSON is longer than 50,000 characters is answered by member A alone,
  because d3-ultra's bidirectional state attention would need ~108 GiB. The rule depends on size only, never on an
  answer.
- **Serving:** both members in one process. Member B runs on its own CUDA stream. The block-FP8 matmul is cuBLAS
  block-scaled when the GPU supports it, else a vendored Triton kernel; either way it is checked at load.
- **Never:** truncation, option dropping, per-benchmark prompts, labels, retries on a wrong answer.

## Declared limits

- Member A refuses (`Unsupported`) a question prompt over 131,072 tokens or with more than 255 options; that refuses
  the request. Nothing is truncated.
- Member B's limit is 131,072 tokens.
- Requests over 50,000 characters skip member B.
- 0 requests were refused.

## How the run was made

- **Suite:** rebuilt with the kit at `9eb2dbe` and verified byte-identical (as for Logits Sigma-1).
- **Run:** a fresh run with the kit's runner at `25c6ec7`. Rows were assigned to 8 shards by estimated cost, with one
  engine process per H200, one request at a time per process, and `--compact`. The shards were concatenated (one
  result per run id) and scored once. Nothing failed and nothing was retried: each of the eight shards loaded once and ran to the end.
- **Engine:** installed from the model repo's `code/` at the published revision.

## Training data, selection and disclosure

- **Member A's LoRA** was trained by Logits in two stages on google/gemma-4-31B-it, then merged into
  KnowLine-31B-Gen1 at scale 0.5.
  - v6, the adapter of [Logits Sigma-1](https://huggingface.co/logitshq/Logits-Sigma-1/tree/5ee69b5ace03c6bdd4e3391ac722993baa7c8d2d), trained on 476,186 rows. v7 warm continued it on 158,750 rows: 52,167 new
    teacher rows and 106,583 rows replayed from v6's file.
  - The data: tasksource typed decisions (411 sources tagged for commercial use); Decision Index benchmark train
    splits in the suite's request format (`ditrain`, 38,278 rows in v6); our generators over public US records (NHTSA,
    OSHA, FJC, openFDA, CVE / CWE / CISA KEV, USPTO) and the Public Jira Dataset; and teacher rows written and
    labelled on Logits' own fleet (Kimi K3.1 Pro, DeepSeek V4.1 Flash).
  - No text or label was written by a Claude or OpenAI model.
  - Every generator and teacher output was decontaminated by 13-gram and exact match against the rebuilt 0.3 suite
    rows.
- **Licence disclosure:** Logits' part is Apache-2.0. The LoRA's data includes:
  - CC BY-SA sources: ARC, SQuAD, DBpedia abstracts, Mr. TyDi Wikipedia passages, BIRD and 72 tasksource sources;
  - 6,133 rows of v6 (1.3%) from 12 sources: GPL / AGPL / ODbL / MPL-labelled tasksource sources, and
    blog_authorship_corpus.

  The model card names them. `DATA_SOURCES.md` in the model repo lists every tasksource source with its licence and
  rows.
- **KnowLine-31B-Gen1** (PelaAI, Apache-2.0) gives member A its starting weights. Its card discloses its data only by
  example and says it trains on Decision Index train splits in request format.
- **d3-ultra** (vLLM Semantic Router Team, Apache-2.0): its training data and method are not disclosed by its
  authors.
- **Selection:** the members, the 50 / 50 weights and the size rule were chosen on a fixed 10% subset of the public
  suite (14,956 requests) and on two held-out decision sets, by rules fixed before the GPU runs. No private test was
  used.

Credits: PelaAI (KnowLine), the vLLM Semantic Router Team (d3-ultra) and Google (Gemma 4). Details are in the model
card's Credits section and in `NOTICE`.
