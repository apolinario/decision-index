# RSI-Jev v5.0-VL 3B

[shgao/rsi-jev-v5.0-vl-3b](https://huggingface.co/shgao/rsi-jev-v5.0-vl-3b) is the first 20 of Qwen3.5-4B-Base's 32 layers, fine-tuned, with a cross-attention scorer over the options: 3.25B parameters including the vision tower, which this suite does not use. It reads the state, the question and every option in one forward pass and returns a probability for each option; it generates nothing.

| | |
|---|---|
| Decision Index 0.2.1 | 38.38 (raw 53.33) |
| Requests | 150,759, all answered, none unsupported, no errors |
| Median request time | 19 ms (RTX PRO 6000, one request at a time) |

## Run

```sh
pip install "rsi-jev[fast] @ git+https://github.com/Shanghua-Gao/RSI-Jev@c6b72b09f263a5f90cec0db94c109a3f6fc6b303"
rsi-jev serve v5.0-vl-3b --port 8000          # 32,768-token and 5,120-option caps; longer is a 422, never cut
python -m decision_index pipeline --engine http --option base_url=http://127.0.0.1:8000 \
    --option model=jev-latest --out runs/RSI-Jev-v5.0-VL-3B
```

The checkpoint is self-contained (tokenizer, embeddings and the 20 layers it runs); nothing is fetched from the base model. A forward pass is capped at 32,768 padded tokens (`RSIJEV_FORWARD_MAX_TOKENS`), so the run fits in one 96 GB card.

## Notes

- **No truncation.** A question longer than 32,768 tokens would get a 422 ("maximum context length") and be recorded as unsupported; none was.
- **Structured criteria** reach the model as compact JSON, as in our v4.0-VL submission. The prompt is the same for every benchmark.
- **Pre-release build.** The run used a pre-release build of the server released at `c6b72b0`. It switched the model's own-token option readout on with `RSIJEV_OPTION_POOL_OWN_TOKENS=1`; the released server reads the same setting from the checkpoint's `meta.json`. The scorer and calibration files are the released ones; the tower came from the fp32 training files and was served in bf16, which gives bitwise the same outputs as the released bf16 tower.
- **One pass, unedited.** No resume, no error rows; `results.jsonl.gz` is the runner's output as written. Re-scoring it with the kit at `87d4650` gives 38.38.
