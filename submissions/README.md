# Model submissions

| Model | Board | Public score | Results | Engine / kit | Hardware | Settings |
|---|---|---|---|---|---|---|
| [vera-large-2o](https://huggingface.co/scar-ai/vera-large-2o/tree/15616f419c4e4d1abf2be250643aa972640b9fd9) | Text | 69.35 | [scores.json](https://huggingface.co/datasets/scar-ai/decision-index-results/blob/f8452fb453a1ea5063ef9654e7ee0be70b27de14/runs/vera-large-2o/scores.json) | `http` → `vera-serve`; kit `25c6ec7` | 1× AMD MI300X | see below |
| [vera-large-2o](https://huggingface.co/scar-ai/vera-large-2o/tree/15616f419c4e4d1abf2be250643aa972640b9fd9) | **Vision** | 76.85 | [vision-scores.json](https://huggingface.co/datasets/scar-ai/decision-index-results/blob/f8452fb453a1ea5063ef9654e7ee0be70b27de14/runs/vision-vera-large-2o/vision-scores.json) | `decide` → `vera-serve`; kit `25c6ec7` | 1× AMD MI300X | see below |
| [vera-medium-omni](https://huggingface.co/scar-ai/vera-medium-omni/tree/1f23a6cf5304e8e53b7ded53980765d5f8f4047c) | Text | 62.17 | [scores.json](https://huggingface.co/datasets/scar-ai/decision-index-results/blob/f8452fb453a1ea5063ef9654e7ee0be70b27de14/runs/vera-medium-omni/scores.json) | `http` → `vera-serve`; kit `25c6ec7` | 1× AMD MI300X | see below |
| [vera-medium-omni](https://huggingface.co/scar-ai/vera-medium-omni/tree/1f23a6cf5304e8e53b7ded53980765d5f8f4047c) | **Vision** | 72.84 | [vision-scores.json](https://huggingface.co/datasets/scar-ai/decision-index-results/blob/f8452fb453a1ea5063ef9654e7ee0be70b27de14/runs/vision-vera-medium-omni/vision-scores.json) | `decide` → `vera-serve`; kit `25c6ec7` | 1× AMD MI300X | see below |

## vera-large-2o and vera-medium-omni settings

Server: `vera-s1` 0.1.7 (wheel in each model repo's `dist/`), one model per GPU:

```bash
vera-serve --model scar-ai/vera-large-2o --engine vllm --no-caps --max-model-len 131072 --port 8201
vera-serve --model scar-ai/vera-medium-omni --engine vllm --no-caps --max-model-len 131072 --port 8202
```

Text: `pipeline --engine http --option base_url=http://127.0.0.1:8201 --option model=vera-large-2o` (port 8202 / `model=vera-medium-omni` for medium).
Vision: `vision pipeline --engine decide --option protocol=decisions --option path=/v1/systemone --option model=<model> --workers 8`.

- Complete runs: text 140,178/140,178 scored, vision 12,210/12,210; 0 unsupported, no declared capacity limits beyond 131,072 tokens per request (never reached).
- `--no-caps`: every request is read whole, nothing truncated. Images at native resolution capped at 1,638,400 pixels (the kit default); the model's processor has no lower cap.
- Results files are uploaded in the kit's `--compact` format (`payload` and `raw_output` dropped so suite text is not republished); they re-score to the same numbers. Full files available on request.
- Suite: rebuilt locally and verified against the pinned hashes. Rebuilding with the kit at `25c6ec7` gives a different `selected-rows` hash because of the SGD history fix (2d7791c) made after the rows were frozen; we rebuilt with that one line reverted to get the pinned `b2b56d6f…`. SGD is not counted in 0.3.
- Latency in our files is MI300X wall time and not representative: part of each text run shared the GPU with the other model.
