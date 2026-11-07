# RSI-Jev v7.0-VL 3B

[shgao/rsi-jev-v7.0-vl-3b](https://huggingface.co/shgao/rsi-jev-v7.0-vl-3b) is [RSI-Jev v7.0-VL 4B](https://huggingface.co/shgao/rsi-jev-v7.0-vl-4b) cut at layer 20, with no retraining: its first 20 decoder layers, its layer-20 head as the main head and its layer-16 head as an early exit. 3.30B parameters including the vision tower, which this suite does not use. It reads the state, the questions and every option in one forward pass and returns a probability for each option; it generates nothing. This run uses the default effort (unset): one threshold, 0.95, at layer 16 for multi-question requests; a single question is answered at layer 20.

| | |
|---|---|
| Decision Index 0.3 public index | 60.01 (raw 69.76) |
| Requests | 140,178 of 140,178 scoreable, all answered, none unsupported, no errors |
| Median request time | 19.6 ms (one request at a time, 1× RTX PRO 6000 Blackwell) |

## Run

```sh
pip install "rsi-jev[fast,vision] @ git+https://github.com/Shanghua-Gao/RSI-Jev@e8d59e4"
rsi-jev serve v7.0-vl-3b --port 8000          # effort unset, as run; 32,768-token and 5,120-option caps; longer is a 422, never cut

git clone https://github.com/apolinario/decision-index && cd decision-index && git checkout 9eb2dbe   # edition 0.3
python -m decision_index pipeline --engine http --option base_url=http://127.0.0.1:8000 \
    --option model=jev-latest --out runs/RSI-Jev-v7.0-VL-3B
python -m decision_index score --edition 0.3 --results results.jsonl.gz --out rescore   # the published file
```

The checkpoint is self-contained (tokenizer, embeddings, the 20 layers, the vision tower and the two heads); nothing is fetched from the base model. The package's `meta.json` turns on the input format it was trained with (`serve_env`); the server applies it at load, so nothing needs to be set. Leave `effort` out of the request and `RSIJEV_EFFORT` unset to get the default this run used.

## Notes

- **How the run was made.** The 0.2.1 suite (150,759 requests) ran with the kit at `87d4650` in seven parts against the same server and checkpoint, and the 2,638 rebuilt GSM8K requests ran under 0.3, as the 0.3 README describes for a resumed 0.2.1 run; all eight parts on 1× RTX PRO 6000 Blackwell. `results.jsonl.gz` is the runner rows of those runs, concatenated and unedited; the old GSM8K and ForecastBench rows are in it and the 0.3 scorer skips them. `environment.json` lists each part's own record.
- **Checkpoint and code.** The run was made before the upload. Every weight, calibration, config and tokenizer file it served is byte-identical to the Hugging Face repo at tag `v7.0-VL-3B` (commit `a2f585ae`); `meta.json` differs in the published parameter count and one descriptive note, which serving does not read. It ran RSI-Jev at `479343b`; the 3B release merge `e8d59e4` adds only the `v7.0-vl-3b` alias to the serving code.
- **Deterministic serving.** The server fixes the configurations of six fla kernels instead of autotuning them at startup (every part logged "tune pin: 6 of 6 kernels"); two server processes give bitwise identical outputs on the RTX PRO 6000 and the H200, so a re-run should reproduce these answers.
- **No truncation.** A question longer than 32,768 tokens would get a 422 ("maximum context length") and be recorded as unsupported; none was.
- **Scores.** `score03` on `results.jsonl.gz` (0.3 scoring, the same code as the released 0.3 kit) gives 60.01, `"complete": true`, 140,178 completed.
