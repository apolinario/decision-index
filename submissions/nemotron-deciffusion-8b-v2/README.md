# Nemotron Deciffusion 8B v2

Self-run Decision Index 0.3 public score **53.89** (raw 65.92, breadth 52.01). The run is complete: 140,620 requests run,
140,178 scored after the official 442 exclusions, 140,178 answered, 0 unsupported, 0 abstained, 0 errors. This is not the
private Full score or a maintainer-verified rank.

- Model: [nemotron-deciffusion/Nemotron-Deciffusion-8B-v2 @ 1dea380](https://huggingface.co/nemotron-deciffusion/Nemotron-Deciffusion-8B-v2/tree/1dea38015e86e101eb978a33fa493818fc71e366) (public, BF16 weights), an 8B
  diffusion language model that reads each typed decision in one bidirectional forward pass.
- Results: [https://huggingface.co/datasets/nemotron-deciffusion/decision-index-0.3-results/tree/08c83d107f2bc8aa9f4f12847d4d9aae4b211c84/runs/nemotron-deciffusion-8b-v2](https://huggingface.co/datasets/nemotron-deciffusion/decision-index-0.3-results/tree/08c83d107f2bc8aa9f4f12847d4d9aae4b211c84/runs/nemotron-deciffusion-8b-v2)
  (run files as written by the kit's runner and scorer, unmodified).
- Kit: apolinario/decision-index @ 9eb2dbe2a358004c8782c66e40a83ac07b953fec, unchanged runner, scorer and `http` engine.
- Server: the released container `ghcr.io/pst2154/nemotron-deciffusion:8b-v2`
  (multi-arch index `sha256:944c0af4967ebd1e461a3b1faa9d16db197288af24393b21c917617461f35276`), `vllm serve` plus a
  `/v1/systemone` adapter. Code: https://github.com/pst2154/nemotron-deciffusion.

## Settings

Start the server (it downloads the pinned checkpoint on first start; works with no network, a read-only root and a non-root
user when the checkpoint is mounted at `/models/checkpoint`):

```bash
docker run --gpus all -p 8770:8770 -v deciffusion-models:/models ghcr.io/pst2154/nemotron-deciffusion:8b-v2
```

Run the suite:

```bash
python -m decision_index run --engine http --option base_url=http://127.0.0.1:8770 \
  --option model=Nemotron-Deciffusion-8B-v2 --compact --out runs/nemotron-deciffusion-8b-v2
```

- **Precision:** FP8 (the container default, `QUANTIZATION=fp8`, dynamic per-tensor) on BF16 weights.
- **Read-out:** one bidirectional forward pass per request; every option's probability is read from the model's answer
  distribution. No sampling, no generated text.
- **Long states:** the container's default long-context path applies. When a prompt is longer than 8,192 tokens, the state
  text is reduced to the 8,192 most question-relevant tokens (relevant sentences for prose, head and tail for traces) before
  the single pass. By the kit's `proxy_tokens` estimate, about 1,319 of the 158,028 rows are above that length (ToolRet 795,
  BRIGHT 439, ACOS 79, ContractNLI 5, BFCL 1). Set `COMPACT_OVER=0` to turn it off.
- **Capacity limits:** none hit. 0 unsupported requests; nothing rejected.

## Run identity

- **Shards:** the 8 `full-rows` shards and `gsm8k-rows` ran as 9 kit runs, spread over four identical FP8 servers on
  one 4-GPU NVIDIA GB200 node (one server per GPU). The shard result lines were concatenated unchanged into one
  `results.jsonl` and scored with the unchanged scorer.
- **Latency in this run** (kit timing, several shards per server at once): median 51.5 ms, mean 80.2 ms, p95 266.5 ms.

## Training data overlap

Training rows were checked against the Decision Index 0.3 public suite before training: a row was rejected if it is an exact
copy of a suite item or shares at least half of its normalized text with one. No benchmark test split was used for
training.
