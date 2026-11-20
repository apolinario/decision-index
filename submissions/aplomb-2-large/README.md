# Aplomb 2 Large

Decision Index 0.3 public score: **75.66** (self-run; raw 81.33). 140,620 requests run, 140,178 scored
(`complete: true`), 0 unsupported, 0 errors.

- Model: https://huggingface.co/empiriolabsai/aplomb-2-large/tree/ed20392ad09146884c82a21e938fe101d444516b
- Results: https://huggingface.co/datasets/empiriolabsai/aplomb-2-large-decision-index/tree/f8a74734a8d4fe899cd7c6282e41a2ed8c7f4932/runs/aplomb-2-large
- Kit: `apolinario/decision-index` @ `25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d`.
- Engine: `aplomb2_large.engine:LargeEngine`, in-process, from the model repository, option `max_work=20000`
  (compact results).

## Settings

```bash
hf download empiriolabsai/aplomb-2-large --revision ed20392ad09146884c82a21e938fe101d444516b --local-dir aplomb-2-large
pip install -r aplomb-2-large/requirements.txt
pip install --no-deps ./aplomb-2-large/engine ./aplomb-2-large/jeffan/inference
python -m decision_index run --engine aplomb2_large.engine:LargeEngine \
    --option karpo=aplomb-2-large/karpo --option jeffan=aplomb-2-large/jeffan --option max_work=20000 \
    --compact --out runs/aplomb-2-large
```

- An equal-weight probability ensemble of two Apache-2.0 decision models on one GPU, each read with its own
  published engine: [karpo-decision-31B](https://huggingface.co/MachinePulseAI/karpo-decision-31B) @ `b8e6fd74`
  (`karpo_decision.serve.engine:DecisionIndexEngine`, `fp8=true`) and
  [Jeffan-27B-1211](https://huggingface.co/tattrongvu/Jeffan-27B-1211) @ `f0ce7043` (`jeffan_engine:DecisionEngine`,
  decoder projections in FP8 at load). Both members' files are unchanged in the model repository. 58.17B parameters
  in total.
- Mixing: for a choice question, the mean of the members' option probabilities (its argmax is the answer); for a
  yes/no question, the mean P(true); for a score question, the mean level probabilities. The weights are fixed and
  never depend on the benchmark.
- Size rule: a request whose `proxy_tokens` times its number of questions exceeds 20,000 is answered by
  karpo-decision-31B alone; a request one member declares it cannot read is answered by the other. Nothing is
  truncated and no option is dropped.
- Declared limits: karpo-decision-31B refuses a question that does not fit 32,768 tokens with its state;
  Jeffan-27B-1211 refuses prompts above 65,536 tokens and reads text only. Every one of the 140,620 requests was
  answered.
- Scored run: the suite split into 8 disjoint shards by `sha256(run_id) mod 8` with the kit's own `Suite` (verified
  strictly) and runner, one engine per GPU; the shards' results are merged unchanged.
- Latency (our measurement, one RTX PRO 6000 Blackwell, one request at a time, 750 rows sampled across the suite by
  benchmark and length): median 137 ms, mean 476 ms, p80 438 ms; peak GPU memory about 83 GB.
