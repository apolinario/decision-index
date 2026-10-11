# Bobcat 1.3

Self-run Decision Index 0.3 public score **59.31** (raw 69.08, breadth 58.38). Complete: 140620 requests all ok; 140178 scored after the official 442 exclusions. No errors, unsupported requests or abstentions in the published results. This is not the private Full score or a maintainer-verified rank.

- Model: https://huggingface.co/sanghwa-na/bobcat-1.3/tree/6bfa8035ec85ac777df910307d20b171d97c2a7b (BF16 weights, FP8 at load). The uniform weight average of three rank-16 LoRA fine-tunes of google/gemma-4-31B-it at revision 842da3794eaa0b77d5f08bae87a17459d91ff475, each merged into the weights.
- Results: https://huggingface.co/datasets/sanghwa-na/bobcat-1.3-decision-index-0.3/tree/7cba886739d85f7c349c3c4c035fac3544ba9625/runs/bobcat-1.3
- Kit and HTTP engine: apolinario/decision-index @ 9eb2dbe2a358004c8782c66e40a83ac07b953fec; unchanged runner, engine and scorer.
- Serving code: https://github.com/foxl-ai/bobcat at 5268e01, module `bobcat.flash_server`, a `/v1/systemone` server.

## Run identity and packaging

- **Suite.** Rebuilt with `suite rebuild` from the public sources and imported; `suite verify` matches every 0.3 hash (rows b2b56d6f, added rows 7429f3c9, GSM8K 75933489, exclusions 331df32d).
- **Servers.** Eight identical servers ran the card's command on the published weights (downloaded anonymously, SHA256SUMS checked), one NVIDIA B200 each, vLLM 0.30.0.
- **Sharding.** The 140620 rows were dealt into 96 shards balanced by proxy tokens, 12 per server; 96 unchanged sequential kit runner processes each ran their own disjoint rows. Results were merged by `run_id` (a later record replaces an earlier one; an error never replaces a non-error record) and scored by the unchanged scorer. No row was re-run.
- **Packaging.** Kit compact mode (payload and raw_output removed). environment.json drops only rows_path and loaded_seconds. PROVENANCE.json has the server command and hashes.
- **Our single-process latency:** one B200, the same server command, the unchanged kit runner as the only client, a fixed random sample of 1500 suite rows (seed 20261009): median 29.4 ms, mean 79.1 ms, p80 79.4 ms. Our Bobcat 1.2 measurement on one RTX PRO 6000 was median 54.5 ms, mean 168.1 ms, p80 179.8 ms. Please run the independent latency and held-out-answer checks.

## Reproducible serving

As for Bobcat 1.2, with these weights and temperatures:

```bash
$PY -c "from huggingface_hub import snapshot_download as s; s('sanghwa-na/bobcat-1.3', revision='6bfa8035ec85ac777df910307d20b171d97c2a7b', local_dir='bobcat-1.3')"
VLLM_USE_FLASHINFER_SAMPLER=0 $PY -m bobcat.flash_server --engine vllm --model bobcat-1.3 \
  --compiler-model bobcat-1.3/compiler --quantization fp8 \
  --temperature choice=1.007793,noul=0.445449,score=1.023435 \
  --name bobcat-1.3 --release-date 2026-10-11 --max-num-seqs 64 --max-model-len 98368 \
  --schedule all --engine-arg max_num_batched_tokens=16384 --engine-arg attention_backend=TRITON_ATTN \
  --host 127.0.0.1 --port 8000 --local
```

Then `python -m decision_index pipeline --engine http --edition 0.3 --option base_url=http://127.0.0.1:8000 --option model=bobcat-1.3 --suite-dir /path/to/suite-0.3 --compact --out runs/bobcat-1.3` from the pinned kit. The server never generates text; it reads the logits of the offered candidates at the first answer position and applies one temperature per question type. A choice accepts at most 255 options; inputs longer than 98368 tokens are refused with HTTP 422, never truncated (none in this run).

## Provenance, training data and overlap

Bobcat 1.3 averages three LoRA arms trained on (a) Bobcat 1.2's rows **without** the rows from ARC-Easy, ARC-Challenge and BANKING77 (and rows derived from them), and without rows whose text overlaps the Decision Index suite; and (b) up to 54,000 questions built from 17 licensed public datasets of human-written text (MASSIVE, CommonsenseQA, GAP, QASC, MedMCQA, MedQA, ScienceQA, AQuA-RAT, WiCE, VitaminC, FaithDial, IBM argument stance, a news-headline sarcasm set, MAUD, LEDGAR, UNFAIR-ToS, ColBERT humor). None of them is a Decision Index benchmark or its train split.

Every new question was audited against all 140620 suite rows (state, instructions and option texts; NFKC, casefold, letter/digit tokens): exact strings of at least 5 tokens, and word 8-gram coverage in both directions (at least half of the question's 8-grams in the suite, or at least half of one suite row's 8-grams in the question), with grams found in more than 20 suite rows treated as shared templates. 688 questions that met a suite row were removed before training (mostly ContractNLI, HoVer, MMLU, SATA-Bench, BRIGHT and CLINC150 neighbours). Bobcat 1.2's rows were checked the same way on their decoded text and their lineage; 7,732 were dropped. Half of the public suite's items (by a stable hash of catalog id and group id) was used, with six held-out public datasets, to choose among candidate models; the score above therefore includes items used for selection. No Jev output and no JevBench item was used.

Independent project; not affiliated with TypeSafe.
