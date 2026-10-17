# jev-d3-46k — vision

- **Track / edition:** Decision Index Vision 0.3
- **Model:** [tinnel123/jev-d3-46k](https://huggingface.co/tinnel123/jev-d3-46k/tree/d0de482276e3506b10b53e04f971cd3df7e25817)
- **Model revision:** `d0de482276e3506b10b53e04f971cd3df7e25817` (the complete materialized checkpoint)
- **Results:** [tinnel123/decision-index-results](https://huggingface.co/datasets/tinnel123/decision-index-results/tree/5235be8201e123c7cf720e2d53e9ecfb5f38b7d4)
- **Run directory:** [`runs/d3_full_checkpoint_vision11_68_2649/`](https://huggingface.co/datasets/tinnel123/decision-index-results/tree/5235be8201e123c7cf720e2d53e9ecfb5f38b7d4/runs/d3_full_checkpoint_vision11_68_2649)
- **Score report:** [`official_scores/vision-scores.json`](https://huggingface.co/datasets/tinnel123/decision-index-results/blob/5235be8201e123c7cf720e2d53e9ecfb5f38b7d4/runs/d3_full_checkpoint_vision11_68_2649/official_scores/vision-scores.json)
- **Public vision index:** `68.26486972026493`
- **Completion:** 12,210/12,210 rows across all 11 benchmark groups; every row is `ok`; `complete=true`; `missing_rows=0`.
- **Engine:** Custom D3 runtime using the standalone materialized full checkpoint, BF16 backbone and FP32 readout. The 200 LoRA target tensors were materialized into the checkpoint; inference loads one model and performs one forward path. There is no runtime weight merge and no two-model forward.
- **Inference code:** [`d3_engine.py`](https://huggingface.co/tinnel123/jev-d3-46k/blob/d0de482276e3506b10b53e04f971cd3df7e25817/d3_engine.py), [`d3_runtime.py`](https://huggingface.co/tinnel123/jev-d3-46k/blob/d0de482276e3506b10b53e04f971cd3df7e25817/d3_runtime.py), and the [full code directory](https://huggingface.co/tinnel123/jev-d3-46k/tree/d0de482276e3506b10b53e04f971cd3df7e25817).
- **Inference code revision:** `d0de482276e3506b10b53e04f971cd3df7e25817`.
- **Decision Index kit:** [`f5a6e4fd85fed05e77878dfc4966d0568bfb45ec`](https://github.com/apolinario/decision-index/commit/f5a6e4fd85fed05e77878dfc4966d0568bfb45ec).
- **Hardware:** Eight MLU accelerator cards, one evaluator rank per card. The public execution receipt records the card count and 8-rank configuration; it does not record a more specific accelerator model.
- **Dependencies:** `torch>=2.8`, `transformers==5.17.0`, `accelerate>=1.10`, `safetensors>=0.6`, `huggingface_hub>=1.0`, `flash-linear-attention==0.5.2`, `einops`, `pillow>=10`, `torchvision`, and `opencv-python-headless`. The HTTP server additionally uses `fastapi` and `uvicorn`.
- **Image resolution and preprocessing:** native image processing with an effective cap of `1,638,400` pixels (1.6 MP) per image, keeping aspect ratio; images are placed before the text in the user turn. The checkpoint processor uses `Qwen2VLImageProcessorFast`, patch size 16, merge size 2, mean/std `[0.5, 0.5, 0.5]`, and accepts source images up to 16 MP before the effective cap. No image was truncated; requests over the runtime input limit are rejected as unsupported.
- **Recorded image budget:** `effective_image_budget=1638400`.
- **Declared capacity limits:** one request at a time in the HTTP server; up to 255 choice options; source image limit 8,000,000 bytes and 16,000,000 pixels; effective model image budget 1.6 MP; no truncation or option filtering. Video inputs are supported by the published runtime (2 fps, 4–32 frames, 200,704 pixels per frame, 16,384 video tokens per request), although this submission is the Vision image track.

## Public benchmark skills

| Benchmark | Skill |
|---|---:|
| CV-Bench | 74.4209 |
| BLINK | 54.8933 |
| RealWorldQA | 58.9364 |
| CharXiv | 72.6817 |
| InfographicVQA | 92.8962 |
| Mind2Web | 77.5266 |
| Winoground | 80.2500 |
| KIE (CORD + FUNSD) | 91.9853 |
| Moderation (Hateful Memes) | 45.4246 |
| R-Bench-M | 16.9800 |
| MMMU-Pro vision | 32.3104 |

The weighted public vision index is **68.26486972026493**.
## Reproduce

Install the runtime and the Decision Index kit at the pinned commit:

```bash
pip install -e '.[vision]' \
  'torch>=2.8' 'transformers==5.17.0' 'accelerate>=1.10' \
  'safetensors>=0.6' 'huggingface_hub>=1.0' \
  'flash-linear-attention==0.5.2' 'einops' 'pillow>=10' \
  torchvision opencv-python-headless
```

Run the standalone HTTP server from the model repository:

```bash
python d3_server.py \
  --model tinnel123/jev-d3-46k \
  --revision d0de482276e3506b10b53e04f971cd3df7e25817 \
  --device cuda:0 --batch-size 8 --verify fast \
  --host 127.0.0.1 --port 8000
```

Or run the kit directly with the packaged engine:

```bash
PYTHONPATH=/path/to/d3-code python -m decision_index vision run \
  --dir /path/to/work/vision \
  --engine d3_engine:D3Engine \
  --option model=tinnel123/jev-d3-46k \
  --option revision=d0de482276e3506b10b53e04f971cd3df7e25817 \
  --option device=cuda:0 --option batch_size=8 --option verify=fast \
  --out runs/jev-d3-46k
python -m decision_index vision score --results runs/jev-d3-46k/results.jsonl
```

The uploaded public run is the frozen standalone full-checkpoint run above. The complete run files and scorer receipt are retained in the linked Hub dataset for rescoring.

## Checkpoint provenance

The checkpoint manifest is `52c4fc6ceb3e2393a5bf1da788422d84eb72c6c19635470a0e7dcd04623b4fe4`. It materializes the D3 body plus the 46k LoRA delta in BF16 while preserving the D3 readout, embeddings and LM projection. The official standalone rerun used 8 ranks, produced 12,210 successful rows, and was scored with the Vision 0.3 scorer. The public score receipt is [available with the run](https://huggingface.co/datasets/tinnel123/decision-index-results/blob/5235be8201e123c7cf720e2d53e9ecfb5f38b7d4/runs/d3_full_checkpoint_vision11_68_2649/SCORE_RECEIPT.json).
