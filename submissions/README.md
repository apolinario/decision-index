# Decision Index model submissions

| Model | Parameters | Public Index (0.3 suite) | Model / evaluated revision | Complete public run | Engine / hardware |
| --- | ---: | ---: | --- | --- | --- |
| JEV Ettin 150M V16 | 149,610,241 | **37.31** | [Elfsong/jev-ettin-150m-v16](https://huggingface.co/Elfsong/jev-ettin-150m-v16/tree/10e1e58c6a18bef3880f83697c9fb323e3181ee6), `10e1e58c6a18bef3880f83697c9fb323e3181ee6` | [Elfsong/jev-decision-index-v16](https://huggingface.co/datasets/Elfsong/jev-decision-index-v16/tree/c62b567a90936ac5ecdfbb2c61f46815334ac009); [scores.json](https://huggingface.co/datasets/Elfsong/jev-decision-index-v16/blob/c62b567a90936ac5ecdfbb2c61f46815334ac009/runs/jev-ettin-150m-v16/scores.json) | `decision_index_engine:DecisionEngine`, Crivo `7a6c7453c06be576e1b48aeb70429318402791d7`; 8 × NVIDIA B200, one process per GPU |

## JEV Ettin 150M V16

Please review/rescore the public run, evaluate the model on both private components of **Decision Index 0.4.1**, and run the official single-process RTX PRO 6000 latency checks. The **Full score is pending**; 37.31 is the public index only.

**Current board standard (0.4.1):** The [official methodology and October 21 changelog](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/0b5ecaf2396891e76c13a19aa17b874b59abcc37/data/methodology.json) expand and refresh the private new-domain tests. Please evaluate this submission on those current tests. Full score = **60% private same-skill + 40% refreshed private new-domain**. The public panel, area/gold weights, chance baselines and frozen corpus hash match the unchanged 0.3 suite; [verification](https://huggingface.co/datasets/Elfsong/jev-decision-index-v16/blob/c62b567a90936ac5ecdfbb2c61f46815334ac009/standard-0.4.1-verification.json) records the checks against Space revision `0b5ecaf2396891e76c13a19aa17b874b59abcc37`. Evaluated weights, inference code and original predictions are unchanged; the updated dataset revision only adds submission documentation and standard-verification metadata.

**Artifact:** V16 `mix10`, selected step 28,000, **149,610,241 parameters**. A native bidirectional ModernBERT encoder initialized from `cross-encoder/ettin-reranker-150m-v1` through our V15/V16 supervised stages, with a shared [OPT]-marker option readout. Our weights were not initialized from the released Crivo weights. We retain attribution for its training recipe and unchanged inference code.

**Pinned weights/code:** `10e1e58c6a18bef3880f83697c9fb323e3181ee6`. **Pinned results:** `c62b567a90936ac5ecdfbb2c61f46815334ac009`. **Weights SHA-256:** `47728332342de9e46bd2fc6fa5bc154e10b4419d7fdfcdb58695602421c5bca2`.

**Suite:** The full public 0.3 suite accepted by the 0.4.1 board. **140,620 original request records**, **140,178 scoreable** after 442 official exclusions. Every request is `ok`; zero errors or unsupported requests. `scores.json` says `"complete": true`. Public index **37.31**, balanced raw **53.50**, breadth skill **32.84**. The run contains the original scores, index, benchmark summary, environment/status and losslessly compressed compact results. Decompression restores the original prediction bytes; `prediction-integrity.json` records both hashes. No benchmark prompts or training examples are redistributed.

**Code and validation:** [Inference source](https://huggingface.co/Elfsong/jev-ettin-150m-v16/tree/10e1e58c6a18bef3880f83697c9fb323e3181ee6/inference) is byte-identical to Crivo revision `7a6c7453c06be576e1b48aeb70429318402791d7`. Original kit: `f8552a13b643d39f61185f5951bb5fed16bedf51`. Latest kit `25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d` fully rescores the saved run to 37.31, with every benchmark score and index component matching. A deterministic **100-request sample spanning all 43 reported benchmarks** reproduces every answer and probability exactly: maximum probability difference **0.0**. [Validation](https://huggingface.co/datasets/Elfsong/jev-decision-index-v16/blob/c62b567a90936ac5ecdfbb2c61f46815334ac009/runs/jev-ettin-150m-v16/validation.json) and [full rescore proof](https://huggingface.co/datasets/Elfsong/jev-decision-index-v16/blob/c62b567a90936ac5ecdfbb2c61f46815334ac009/runs/jev-ettin-150m-v16/rescore-verification.json).

**Exact settings:** Python 3.11.2; PyTorch 2.8.0+cu128; Transformers 5.19.0. FP32 weights, BF16 CUDA forward autocast, FP32 logits and FP64 softmax, SDPA, temperature 1.0. Three CPU threads/process. One request at a time; at most **32 questions from that request** per microbatch and a **65,536 padded-token budget**. Eight B200 processes use deterministic request shards; results are merged unchanged. No cross-request batching, generation, CoT, ensembling, quantization, fitted temperature or per-benchmark routing.

**Limits:** Text `choice`/`noul` only. **32,768 complete serialized tokens per question**, including the state, instructions and all options. Over-limit inputs raise `Unsupported`; no truncation or candidate filtering. No separate option-count cap; token capacity applies. The training library also supports ordinal scores, but the evaluated adapter does not expose them or vision inference. Training used **8,192 tokens**; the longer evaluation budget uses inherited RoPE handling, not newly trained context extension.

**Training disclosure:** Full-parameter SFT with CE + 0.5 × sum-form Brier; no RL, teacher inference, KD, averaging or post-hoc temperature fitting. The mixture contains Bekko, native training splits and procedural typed decisions; ToolACE/Glaive tools; ACOS; Home/CLadder; HoVer, ContractNLI, RAGTruth, NLI4CT and When2Call replay. [Training provenance](https://huggingface.co/Elfsong/jev-ettin-150m-v16/blob/10e1e58c6a18bef3880f83697c9fb323e3181ee6/training-provenance.json) and [full V15/V16 plans, source census and fingerprints](https://huggingface.co/Elfsong/jev-ettin-150m-v16/tree/10e1e58c6a18bef3880f83697c9fb323e3181ee6/training) are public. Public aggregate results guided research; public targets were not used for updates or within-run checkpoint selection. V16 was chosen for submission after comparing completed public runs. Exact-input filtering cannot rule out semantic/pretraining overlap or domain overlap with unknown private tests; please apply the normal overlap checks.

**Latency:** The saved run reports B200 median **6.5 ms**, mean **11.3 ms**, p95 **22.8 ms**. These concurrent-B200 timings are diagnostic, not the official single-process RTX PRO 6000 measurements or an eligibility claim.

### Reproduce the public run

Use Python 3.11, a CUDA-enabled PyTorch 2.8.0 build and an officially rebuilt/imported public suite. Download the pinned inference package:

```bash
python -m pip install 'huggingface-hub==2.2.0'
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('Elfsong/jev-decision-index-v16', repo_type='dataset', revision='c62b567a90936ac5ecdfbb2c61f46815334ac009',
                  allow_patterns=['inference/*'], local_dir='jev-v16-submission')
PY
cd jev-v16-submission
python -m pip install 'git+https://github.com/apolinario/decision-index.git@25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d'
python -m pip install -r inference/requirements.txt
python inference/reproduce.py --suite /path/to/suite-0.3 --out runs/v16-reproduction
```

The reproduction script downloads the exact model revision above and runs the official resumable pipeline with the evaluated settings. The suite is not redistributed.

### Private evaluation

Download the same pinned model, add `MODEL_DIR/inference` to `PYTHONPATH`, set `torch.set_num_threads(3)`, and call:

```python
from decision_index_engine import DecisionEngine
engine = DecisionEngine(path=MODEL_DIR, device='cuda', batch_size=32,
                        max_length=32768, token_budget=65536)
response, raw = engine(state, questions)
```

`state` and `questions` use the kit schema. Every supplied choice key receives a normalized probability; `noul` returns P(true). No benchmark identifier is needed. Both repositories are public and full inference source is included, so no submitter-hosted service is required.
