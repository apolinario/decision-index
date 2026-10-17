# Herbert v0.5, vision

Weights and serving code: [`leinenterprises/Herbert_v0.5`](https://huggingface.co/leinenterprises/Herbert_v0.5) @ `9b5cb88` (Apache 2.0).
A LoRA on Qwen/Qwen3.8-27B; the code is in the repo's `code/` folder (same server as jev-vision-27b).

## Serve

```bash
hf download leinenterprises/Herbert_v0.5 --revision 9b5cb88 --local-dir Herbert_v0.5
cd Herbert_v0.5/code
pip install vllm==0.31.0 fastapi uvicorn peft accelerate "decision-index[vision] @ git+https://github.com/apolinario/decision-index@f5a6e4f"
python -m train.merge --ckpt .. --out ../merged      # adapter -> plain bf16 model
python -m serve.server --model ../merged --perms 1 --port 8000 --max-model-len 65536 --max-pixels 1638400
```

One forward pass per question, nothing generated. Images arrive as the kit sends them (capped at 1,638,400 pixels) and
the server's own `max_pixels` is the same 1,638,400, so the board's cap is the only one.

## Run and score

```bash
decision-index suite rebuild --vision --work work
decision-index vision pipeline --dir work/vision --engine decide --option base_url=http://127.0.0.1:8000 \
  --option protocol=decisions --option model=herbert-v0.5 --workers 8 --out runs/vision-herbert-v0.5
```
