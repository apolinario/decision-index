# Amos 1 large, vision

Weights and serving code: [`mukaddamzaid/amos-1-large`](https://huggingface.co/mukaddamzaid/amos-1-large) @ `362f4dc`
(Apache 2.0). The serving package is the repo's `serve/` folder; it pins torch 2.14.0, transformers 5.17.0 and
flash-linear-attention 0.5.2.

## Serve

```bash
hf download mukaddamzaid/amos-1-large --revision 362f4dc --local-dir amos-1-large
cd amos-1-large/serve && pip install -e .
python -m amos.serve ../ --device cuda --port 8736 --name amos-1-large \
  --rotations 1 --max-len 131072 --strict --no-state-cache
```

One forward pass per request, nothing generated. Images go in as the kit sends them (capped at 1,638,400 pixels);
the image processor's own limit is 16.7 MP, so the board's cap is the only one.

## Run and score

```bash
pip install "decision-index[vision] @ git+https://github.com/apolinario/decision-index@f8552a1"
decision-index suite rebuild --vision --work work
decision-index vision run --dir work/vision --engine decide --option base_url=http://127.0.0.1:8736 \
  --option path=/v1/systemone --option protocol=decisions --option model=amos-1-large \
  --workers 1 --compact --out runs/amos-1-large-vision
decision-index vision score --dir work/vision --results runs/amos-1-large-vision/results.jsonl \
  --out runs/amos-1-large-vision
```
