# AutoTrust-JEV-27B v1.1 — vision

Self-run Decision Index 0.3 **public vision index 74.33**. Complete: all 12,210 rows of the public vision suite `ok`
(Winoground included), `vision-scores.json` says `"complete": true`, 0 missing rows. This is not the private Full score or
a maintainer-verified rank.

- Model: [autotrust/AutoTrust-JEV-27B-v1.1 @ `ae15690`](https://huggingface.co/autotrust/AutoTrust-JEV-27B-v1.1/tree/ae156900e9f338e423042b89e25d0fd8093064d4),
  fine-tuned from [vllm-sr/d3](https://huggingface.co/vllm-sr/d3) (Apache-2.0), built on Qwen/Qwen3.8-27B @ `1d4bf0f`.
  The vision tower is d3's, unchanged: all 333 visual tensors are byte-identical to vllm-sr/d3. The fine-tuning beyond d3
  was text only, so image decisions are zero-shot on top of d3's vision tower.
- Results: [runs/autotrust-jev-27b-v1.1-vision](https://huggingface.co/datasets/autotrust/jev-decision-index-results/tree/086ec7a0a804eae0f9e27a0fdd7e6501e8cf4777/runs/autotrust-jev-27b-v1.1-vision)
  (`results.jsonl`, `vision-scores.json`, `environment.json`, `status.json`, `latency.json`, `vision-verify.json` and the
  code that ran in `source/`). `vision score` on `results.jsonl` reproduces 74.33.
- Kit: apolinario/decision-index @ `f5a6e4fd85fed05e77878dfc4966d0568bfb45ec`, unchanged vision runner and scorer.
- Engine: `jev_vllm_vision_engine:JevVllmVision` (this directory), a subclass of the kit's `vllm-vl` engine with the
  model's own prompt and code read-out. The model repository's own engine is text only.
- Image resolution: native, capped at 1,638,400 pixels (1.6 MP) by the kit (larger images downscaled, LANCZOS); the
  model's processor limit (`longest_edge` 16,777,216 pixels) is above the cap, so nothing lower was in force.
  `effective_image_budget`: `max_image_pixels` 1,638,400, `max_image_tokens` 3,240 (most image tokens in one request).

| benchmark | rows | accuracy | skill |
|---|---:|---:|---:|
| CV-Bench | 2,038 | 86.1 | 77.1 |
| BLINK | 961 | 73.9 | 59.6 |
| RealWorldQA | 589 | 82.0 | 70.6 |
| CharXiv | 532 | 85.5 | 80.7 |
| InfographicVQA | 244 | 98.8 | 98.4 |
| Mind2Web | 973 | 87.9 | 83.8 |
| Winoground | 800 | 93.2 | 86.5 |
| KIE (CORD+FUNSD) | 2,465 | 95.6 | 92.7 |
| Moderation (Hateful Memes) | 1,213 | 72.2 | 44.4 |
| R-Bench-M (weight 0.5) | 665 | 48.3 | 37.9 |
| MMMU-Pro vision (weight 0.25) | 1,730 | 54.6 | 47.9 |
| **Public vision index** | **12,210** | | **74.33** |

## Reproduce

Linux, one NVIDIA GPU with 80 GB or more (ran on one B200), Python 3.12, torch 2.13 + CUDA 13.2, vLLM 0.31.0,
transformers 5.17.0, [StartLux-Decision](https://github.com/StartLuxLabs/StartLux-Decision) @ `cb1d11f` (Apache-2.0,
for its vLLM prompt-bidirectional plugin).

```bash
pip install vllm==0.31.0 transformers==5.17.0 \
  "startlux-decide @ git+https://github.com/StartLuxLabs/StartLux-Decision@cb1d11f1ac9e9b43b95ef3a6468fa29b607c40fd"
pip install -e '.[vision]'                             # this kit
decision-index suite rebuild --vision --work work      # Winoground terms accepted on the Hub
decision-index vision verify --dir work/vision
hf download autotrust/AutoTrust-JEV-27B-v1.1 --revision ae156900e9f338e423042b89e25d0fd8093064d4 \
  --local-dir AutoTrust-JEV-27B-v1.1
PYTHONPATH=submissions/autotrust-jev-27b-v1.1-vision decision-index vision pipeline --dir work/vision \
  --engine jev_vllm_vision_engine:JevVllmVision --option model=AutoTrust-JEV-27B-v1.1 --chunk 256 \
  --out runs/autotrust-jev-27b-v1.1-vision
```

## Suite

Rebuilt with `suite rebuild --vision` at kit `f5a6e4f`; `vision verify`: all 12 source sets and all 11 benchmarks match
the board's pins (item counts, items and image digests, sorted run ids, payload hashes), `"ok": true`
(`vision-verify.json`). The Mind2Web box labels were drawn with Arial Bold (`--font`), as the verify pins require.

## Inference

- **Prompt.** d3's image-prompt layout with the model's prompt strings (`jev_format.py`, unchanged from d3's): system =
  the model's system prompt; user = one image placeholder per image, in the row's order, then the model's text (state,
  question, options under single-token answer codes `A`, `B`, ..., closing line); the assistant turn opened with the empty
  think block. For one image this is byte-identical to d3's own `image_messages` prompt.
- **Read-out.** One decoding step restricted to the question's first n answer-code tokens (`vllm_readout.json`),
  processed log-probabilities, softmax at temperature 1.0. Nothing generated beyond that one token; no abstention.
  The row's `prompt.readout` prefix (`The answer is (`, R-Bench-M and MMMU-Pro vision) is not used: the model reads
  its answer at its own read-out position on every row. MMMU-Pro rows (`options_in_image`) list the letters as options.
- **vLLM.** In process, configured like the model repository's `serve_vllm.sh`: StartLux prompt-bidirectional plugin on
  (`STARTLUX_PREFIX_BIDIR=1`), bf16, `max_model_len` = `max_num_batched_tokens` = 65,536, no prefix caching, no chunked
  prefill, FLASH_ATTN, processed log-probabilities (up to 255), up to 8 images per request. `max_num_seqs` 8: the
  plugin needs every prompt prefilled whole in one step, and vLLM schedules only the text before an image when the
  image does not fit in its encoder cache (sized to `max_num_batched_tokens`); 8 sequences always fit. Scheduling only;
  it does not change any answer.

## Latency

`latency.json`: one row per engine call (vision runner, chunk 1), the run's engine and settings, one B200, 220 rows
(20 per benchmark): median **277 ms**, mean **388 ms**, p80 **432 ms** per request, image loading and prompt building
included (p95 971 ms). The `total_wall_ms` values in `results.jsonl` are chunk wall time divided by the chunk's rows.

## Capacity

Up to 8 images per request (the suite has at most 4), up to 255 options per question; context 65,536 tokens; nothing
truncated, no row refused.

## Training data

Fine-tuned from vllm-sr/d3 on TypeSafe Jev teacher distributions, ground-truth decision data built from public datasets
(including public training splits of datasets whose test splits the suite uses) and rows of the public
`tasksource/tasksource-jev-typed-decisions`; text only. No test split of any suite benchmark was used. The vision tower
and d3's own training data are d3's.

Hardware: 1 × NVIDIA B200 (183 GB). Not affiliated with TypeSafe AI.
