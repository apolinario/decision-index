# RSI-Jev v6.1-VL 27B

[shgao/rsi-jev-v6.1-vl-27b](https://huggingface.co/shgao/rsi-jev-v6.1-vl-27b) is Qwen3.8-27B, fine-tuned, with decision heads after layers 48, 56 and 64; 26.66B parameters including the vision tower, which this suite does not use. Its weights are the uniform average of two fine-tunes of the same base, with the temperatures refit. It reads the state, the questions and every option in one forward pass and returns a probability for each option; it generates nothing. It chooses its depth: a question is answered at the first of the three layers that is confident enough. This run uses effort `auto`, set as the server default: every text request, one question included, is answered at layer 48 if the calibrated top-1 probability reaches 0.8, else at layer 56 if it reaches 0.7, else at layer 64. The other configuration, effort unset, reads 65.46.

| | |
|---|---|
| Decision Index 0.3 public index | 65.64 (raw 73.93) |
| Requests | 140,178 of 140,178 scoreable, all answered, none unsupported, no errors |
| Request time | median 45.5 ms, mean 120.2 ms, 80th percentile 138.1 ms (one request at a time, 1× NVIDIA H200) |

## Run

```sh
pip install "rsi-jev[fast,vision] @ git+https://github.com/Shanghua-Gao/RSI-Jev@fc982bbd3fb185af2ce629091588cb640400dafb"
rsi-jev serve v6.1-vl-27b --effort auto --port 8000   # or RSIJEV_EFFORT=auto; 32,768-token and 5,120-option caps; longer is a 422, never cut

git clone https://github.com/apolinario/decision-index && cd decision-index && git checkout 9eb2dbe   # edition 0.3
python -m decision_index pipeline --engine http --option base_url=http://127.0.0.1:8000 \
    --option model=jev-latest --out runs/RSI-Jev-v6.1-VL-27B
python -m decision_index score --edition 0.3 --results results.jsonl.gz --out rescore   # the published file
```

The checkpoint is self-contained (tokenizer, embeddings, the 64 layers, the vision tower and the three heads); nothing is fetched from the base model. It runs in bf16 on one 96 GB GPU (84.3 GB peak on an RTX PRO 6000 at the 32,768-token cap). Start the server with `--effort auto` (or `RSIJEV_EFFORT=auto`) and leave `effort` out of the request to get what this run used; the per-layer thresholds come from `meta.json` (`adaptive.auto_thresholds`). A request's own `effort` overrides the server default.

## Notes

- **How the run was made.** The 0.2.1 suite (150,759 requests) ran with the kit at `87d4650` in seven parts against the same server and checkpoint, each on 1× NVIDIA H200: the 16,000-request part and six shards. The 2,638 rebuilt GSM8K requests ran with the kit at `62d2f51`, also on 1× NVIDIA H200, as the 0.3 README describes for a resumed 0.2.1 run. `results.jsonl.gz` is the runner rows of those runs, concatenated and unedited; the old GSM8K and ForecastBench rows are in it and the 0.3 scorer skips them. `environment.json` lists each part's own record.
- **Serving path.** The server right-pads text inputs to a multiple of 64 tokens and replays CUDA graphs for short inputs; `meta.json` (`serving.pad_multiple`) sets this, and the model's numbers are defined on that path. Every part logged it as on.
- **Checkpoint.** The run served the release package before upload; every part logged the same `scorer.safetensors` and `meta.json` checksums and passed a check of every file in the package's `md5.txt`. Before upload, meta.json was reduced to the keys the loader reads (provenance and internal paths removed); weights, heads and calibration are byte-identical to the gated package.
- **Code.** The run served from RSI-Jev `0c18400`; the commands above install current main, `fc982bb`, which has changed only documentation since.
- **No truncation.** A question longer than 32,768 tokens would get a 422 ("maximum context length") and be recorded as unsupported; none was.
- **Hardware.** On 1× RTX PRO 6000 Blackwell, single process, one request at a time, the default (unset) effort on the 16,000 requests of the kit's stratified 16,000-request sample took median 105.7 ms, mean 379.4 ms, 80th percentile 445.8 ms, with 84.3 GB peak GPU memory. Effort auto exits earlier but was not timed on this GPU.
- **Scores.** Re-scoring `results.jsonl.gz` with the kit at `62d2f51` (`score --edition 0.3`; the released edition 0.3, `9eb2dbe`, has identical code) gives 65.64, `"complete": true`, 140,178 completed.
