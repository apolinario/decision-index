# Amos 1 large

Weights and serving code: [`mukaddamzaid/amos-1-large`](https://huggingface.co/mukaddamzaid/amos-1-large) @ `362f4dc`
(Apache 2.0). The serving package is the repo's `serve/` folder; it pins torch 2.14.0, transformers 5.17.0 and
flash-linear-attention 0.5.2.

## Serve

```bash
hf download mukaddamzaid/amos-1-large --revision 362f4dc --local-dir amos-1-large
cd amos-1-large/serve && pip install -e .
python -m amos.serve ../ --device cuda --port 8000 --name amos-1-large \
  --rotations 2 --max-len 131072 --strict --no-state-cache
```

Nothing is generated. `--rotations 2` reads each choice question twice, with its options in two orders, and averages the two; the two readings run as one batched forward pass when they fit in 65,536 tokens together, and one after the other above that. Requests over 16K tokens
are read a layer at a time in 16K chunks, which gives the same answers as one pass within bf16 rounding and keeps
memory flat; requests over 131,072 tokens get a 422.

## Run and score

```bash
python -m decision_index run --engine http --option base_url=http://127.0.0.1:8000 --option model=amos-1-large \
  --edition 0.3 --suite-dir suite-0.3 --out runs/amos-1-large-0.3 --compact
python -m decision_index score --results runs/amos-1-large-0.3/results.jsonl --edition 0.3 --suite-dir suite-0.3 \
  --engine amos-1-large --out runs/amos-1-large-0.3
```

Our run split the rows into four shards on four H100s, one server and one client per shard, and concatenated the
results before scoring.
