# jev-vision-27b, vision

Weights and serving code: [`e12ex2/jev-vision-27b`](https://huggingface.co/e12ex2/jev-vision-27b) @ `483df76` (Apache 2.0).
A LoRA on Qwen/Qwen3.8-27B; the code is in the repo's `code/` folder.

## Serve

```bash
hf download e12ex2/jev-vision-27b --revision 483df76 --local-dir jev-vision-27b
cd jev-vision-27b/code
pip install vllm==0.31.0 fastapi uvicorn peft accelerate "decision-index[vision] @ git+https://github.com/apolinario/decision-index@f5a6e4f"
python -m train.merge --ckpt .. --out ../merged      # adapter -> plain bf16 model
python -m serve.server --model ../merged --perms 1 --port 8000 --max-model-len 65536 --max-pixels 1638400
```

One forward pass per question, nothing generated: the answer is read from the option-label logits at the answer
position and softmaxed over the listed options. Images arrive as the kit sends them (capped at 1,638,400 pixels) and
the server's own `max_pixels` is set to the same 1,638,400, so the board's cap is the only one.

## Run and score

```bash
decision-index suite rebuild --vision --work work
decision-index vision pipeline --dir work/vision --engine decide --option base_url=http://127.0.0.1:8000 \
  --option protocol=decisions --option model=jev-vision-27b --workers 8 --out runs/vision-jev-vision-27b
```
