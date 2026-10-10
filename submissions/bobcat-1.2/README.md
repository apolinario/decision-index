# Bobcat 1.2

Self-run Decision Index 0.3 public score **58.67** (raw 68.55, breadth 57.84). Complete: 140620 requests all ok; 140178 scored after the official 442 exclusions. No errors, unsupported requests or abstentions in the published results. This is not the private Full score or a maintainer-verified rank.

- Model: https://huggingface.co/sanghwa-na/bobcat-1.2/tree/7d212e7bb3d4ed1e0106219a845a1a48ab8735f4 (BF16 weights, FP8 at load). A LoRA (rank 16) fine-tune of google/gemma-4-31B-it at revision 842da3794eaa0b77d5f08bae87a17459d91ff475, merged into the weights.
- Results: https://huggingface.co/datasets/sanghwa-na/bobcat-1.2-decision-index-0.3/tree/aef7f3cb9845bed8cceddcfab6ffdc2ce0cfd82f/runs/bobcat-1.2
- Kit and HTTP engine: apolinario/decision-index @ 9eb2dbe2a358004c8782c66e40a83ac07b953fec; unchanged runner, engine and scorer.
- Serving code: https://github.com/foxl-ai/bobcat at bcd7e33eea05299c56aac77dbbce4f617b9fae79, module `bobcat.flash_server`, a `/v1/systemone` server.

## Run identity and packaging

- **Suite.** Built with `suite rebuild` from the public sources, then imported. `suite verify` matches every 0.3 hash: rows b2b56d6f, added rows 7429f3c9, GSM8K 75933489, exclusions 331df32d.
- **Servers.** Two identical servers ran the same command and weights. Each had one NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB) and vLLM 0.30.0.
- **Sharding.** The 140620 rows were dealt round-robin into 12 shards. Each shard was cut once, at the row where its remaining rows' proxy tokens balance. Server 1 ran the first part of every shard (43522 rows) and server 2 the rest (97098 rows).
- **Runners and scoring.** Twenty-four unchanged sequential kit runner processes, 12 per server, each ran its own disjoint rows. The results were merged by `run_id` and scored by the unchanged scorer.
- **One interruption.** Server 1's server process was stopped for about three minutes by an operator error. 36620 of its rows got connection errors in that window. The kit runner's resume re-sent exactly those rows to the restarted server (same command, same weights), and all of them returned ok. No other row was re-run.
- **Merge rule.** The merge keeps one record per `run_id`. As in the kit's loader, a later record replaces an earlier one. An error never replaces a non-error record. PROVENANCE.json describes all of this, with hashes.
- **Packaging.** Published results are the kit's compact mode: payload and raw_output are removed, and every other field is as the runner wrote it. Re-scoring them with the pinned kit reproduces 58.67.
- **Environment.** environment.json removes only the local shard paths (rows_path) and loaded_seconds. The 24 runner environments are otherwise identical.
- **Hash.** Compact uncompressed results SHA-256: c748987775a38fc934f618a80d13eb5a5a0d94888ee32796daf69c07990083fc.
- **Loaded HTTP latency** in this run, with 12 concurrent clients per server: median 611.5 ms, mean 2063.2 ms. This is not an eligibility measurement.
- **Our single-process measurement:** one RTX PRO 6000, the same server command, the unchanged kit runner as the only client, on a fixed random sample of 1500 suite rows (seed 20261009). Median 54.5 ms, mean 168.1 ms, p80 179.8 ms. Please run the independent latency and held-out-answer checks.

## Reproducible serving

Linux, one NVIDIA GPU, Python 3.12 via uv; tested on one RTX PRO 6000 Blackwell 96 GB.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"
git clone https://github.com/foxl-ai/bobcat && cd bobcat
git checkout bcd7e33eea05299c56aac77dbbce4f617b9fae79
export UV_PYTHON_PREFERENCE=only-managed
uv venv --python 3.12 .venv-serve
uv pip install --no-config --python .venv-serve/bin/python vllm==0.30.0 fastapi uvicorn scipy jinja2 \
  "tokenizers>=0.21" huggingface_hub
export PYTHONPATH=$PWD/src PY=.venv-serve/bin/python
$PY -c "from huggingface_hub import snapshot_download as s; s('sanghwa-na/bobcat-1.2', revision='7d212e7bb3d4ed1e0106219a845a1a48ab8735f4', local_dir='bobcat-1.2')"
VLLM_USE_FLASHINFER_SAMPLER=0 $PY -m bobcat.flash_server --engine vllm --model bobcat-1.2 \
  --compiler-model bobcat-1.2/compiler --quantization fp8 \
  --temperature choice=1.071833,noul=0.264866,score=1.157635 \
  --name bobcat-1.2 --release-date 2026-10-09 --max-num-seqs 64 --max-model-len 98368 \
  --schedule all --engine-arg max_num_batched_tokens=16384 --engine-arg attention_backend=TRITON_ATTN \
  --host 127.0.0.1 --port 8000 --local
```

Then, from the pinned kit checkout with a rebuilt, verified suite:

```bash
python -m decision_index pipeline --engine http --edition 0.3 \
  --option base_url=http://127.0.0.1:8000 \
  --option model=bobcat-1.2 \
  --suite-dir /path/to/suite-0.3 --compact --out runs/bobcat-1.2
```

This is the single-client reproduction path, not the two-server, 24-shard topology above. Run the server from the repository root. `--local` turns off the shared secret the server otherwise requires, so bind it to loopback only.

**Inference.** The server builds the model input from state and questions with one fixed renderer, with no per-benchmark prompt changes. It never generates text: it reads the logits of the offered candidates at the first answer position of one forward pass, and the host builds the closed JSON reply from the request's own names. It then applies one temperature per question type: choice 1.071833, noul 0.264866, score 1.157635. A temperature never changes an argmax. The weights are FP8 at load (vLLM online dynamic FP8 of the BF16 weights), with Triton attention, a maximum model length of 98368 tokens, 16384 batched tokens and at most 64 sequences.

**Capacity.** A choice question accepts at most 255 options. Inputs where state plus question is longer than the 98368-token model length are refused. Both are refused with HTTP 422 and never truncated. No request in this run reached a limit (0 unsupported).

**Refusal wording.** The servers in this run differed from the public commit above only in the wording of three capacity-refusal messages. They carry the kit's standard phrases ("options per choice", "longer than the maximum model length"), so the kit counts such refusals as unsupported. At bcd7e33 the same refusals return HTTP 422 with older wording, which the kit counts as errors. Answers are identical either way.

## Provenance, training data and overlap

Bobcat 1.2 is a LoRA fine-tune of google/gemma-4-31B-it, trained on rows sampled from our earlier training files:
- the Bobcat 1.1 training mixture and its derived prompt-injection rows;
- the Bobcat Flash training corpus;
- long product-task rows.

These files include the train splits of BANKING77, ARC-Easy and ARC-Challenge. No Decision Index row, and no test or validation split of a Decision Index benchmark, was used for training, calibration or model selection. No Jev output was used.

We audited overlap between every 0.3 suite row (state plus question instructions; candidate labels left out) and every string of 13 of our training files, which cover every file our released models were trained from. Bobcat 1.2's own rows are stored as token ids, so they were checked through the text files they were sampled from; the counts below are therefore an upper bound for Bobcat 1.2. Text was normalized with NFKC, casefolding and letter/digit tokens. Two checks were run:
- exact strings or lines of at least 5 tokens;
- word 8-gram containment.

Rows with at least half of their 8-grams found in those files:

| Benchmark | Rows | Containment >= 0.5 | Containment = 1.0 |
|---|---:|---:|---:|
| MMLU | 14042 | 34 | 4 |
| ARC-Easy | 2376 | 21 | 1 |
| MMLU-Pro | 12032 | 15 | 3 |
| HoVer | 4000 | 11 | 0 |
| ARC-Challenge | 1172 | 7 | 1 |
| BANKING77 | 3080 | 6 | 0 |
| ANLI | 3200 | 2 | 0 |
| Amazon ESCI | 5000 | 1 | 0 |

That is 97 rows in total. They come from the Bobcat 1.1 training mixture and its derived copies, the Bobcat Flash training corpus, and the long product-task rows. These include the ARC and BANKING77 train splits, which carry questions similar to their test splits.

Exact line matches without containment are mostly in RouterBench, SGD, BRIGHT, ToolRet, PhishNChips and API-Bank. They are shared short lines (tool descriptions, common dialogue turns), not copied items. BANKING77 has 20 exact test utterances; in an earlier check, 18 of them also occur in BANKING77's own train split. GSM8K, HellaSwag, WinoGrande and CLINC150 have no row at 0.5 or above.

Independent project; not affiliated with TypeSafe.
