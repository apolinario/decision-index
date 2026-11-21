# Aplomb 2

Decision Index 0.3 public score: **58.83** (self-run; raw 68.96). 140,620 requests run, 140,178 scored
(`complete: true`), 0 unsupported, 0 errors.

- Model: https://huggingface.co/empiriolabsai/aplomb-2/tree/399027c60602cd832724456b82d24d467ed9cad0
- Results: https://huggingface.co/datasets/empiriolabsai/aplomb-2-decision-index/tree/b375e8344afc2989526fd1b3d68535321a38dbe2/runs/aplomb-2
- Kit: `apolinario/decision-index` @ `9eb2dbe2a358004c8782c66e40a83ac07b953fec` (runner and scorer of the scored run).
- Engine: the kit's `http` engine against the model's own `/v1/systemone` server (`serve_aplomb.py` in the model
  repository), BF16 (compact results).

## Settings

```bash
hf download empiriolabsai/aplomb-2 --revision 399027c60602cd832724456b82d24d467ed9cad0 --local-dir aplomb-2
cd aplomb-2   # Python 3.11
python -m pip install --require-hashes --no-deps --only-binary=:all: \
  --extra-index-url https://download.pytorch.org/whl/cu129 -r requirements.txt
python serve_aplomb.py --model . --port 8000   # POST /v1/systemone, GET /health

# in the kit checkout
python -m decision_index run --engine http --option base_url=http://127.0.0.1:8000 --option model=aplomb-2 \
    --compact --out runs/aplomb-2
```

- A decision model of 5,216,236,436 parameters (Qwen3.5-4B base with the Qwen3-Omni audio encoder), Apache-2.0.
  The repository's three weight sets of that architecture hold 14,329,254,332 parameters in all; any one request is
  read by one set. Requests carry only `model`, `state` and `questions`; the server's defaults are used.
- Declared limits: an input of up to 32,768 tokens, including the longest question, is read in one pass by the
  weights in the repository root. A longer text input, up to 1,000,000 tokens, is read by the weights in
  `long_context/`. A text request of 17 to 32 questions without evidence is read by the weights in `grouped/`. Up to
  128 questions a request and up to 255 options a choice question. A request over a limit is refused, never
  truncated. In the scored run every request was read by the root weights, and no public 0.3 request reaches the
  other two readers.
- Scored run: our server build of the same weights (BF16, NVIDIA RTX PRO 6000 Blackwell), the suite in disjoint
  shards with the kit's own runner and `http` engine; the shards' results are merged unchanged. The published
  package, installed exactly as above, gave the same decision as the scored run on 1,263 of 1,271 questions of 671
  sampled requests (mean absolute probability difference 0.0045).
- Latency (our measurement with the package above, one RTX PRO 6000 Blackwell Server Edition, one request at a
  time, 750 rows sampled across the suite by benchmark and length): median 48 ms, mean 145 ms, p80 126 ms.
- Training overlap: the training data included the `train` splits of WinoGrande and ContractNLI; training rows
  matching an item of the public suite were removed. The model card's "Evaluation data exposure" section lists the
  rest.
