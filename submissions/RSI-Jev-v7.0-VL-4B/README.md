# RSI-Jev v7.0-VL 4B

[shgao/rsi-jev-v7.0-vl-4b](https://huggingface.co/shgao/rsi-jev-v7.0-vl-4b) is fine-tuned from Qwen3.5-4B (the instruction-tuned model), with decision heads after layers 16, 20 and 32; 4.69B parameters including the vision tower, which this suite does not use. It reads the state, the questions and every option in one forward pass and returns a probability for each option; it generates nothing. Its input format is new: the option list is repeated once and read from the second copy, and on the full-attention layers the prompt and the options attend to each other in both directions. It chooses its depth: a question is answered at the first of the three layers that is confident enough. This run uses the default effort (unset): one threshold, 0.95, at layers 16 and 20 for multi-question requests; a single question is answered at layer 32.

| | |
|---|---|
| Decision Index 0.3 public index | 60.61 (raw 70.21) |
| Requests | 140,178 of 140,178 scoreable, all answered, none unsupported, no errors |
| Median request time | 29.1 ms (one request at a time, 1× RTX PRO 6000 Blackwell) |

## Run

```sh
pip install "rsi-jev[fast,vision] @ git+https://github.com/Shanghua-Gao/RSI-Jev@479343b27946e2a44a8bb72e9fc4dc09a824cf20"
rsi-jev serve v7.0-vl-4b --port 8000          # effort unset, as run; 32,768-token and 5,120-option caps; longer is a 422, never cut

git clone https://github.com/apolinario/decision-index && cd decision-index && git checkout 9eb2dbe   # edition 0.3
python -m decision_index pipeline --engine http --option base_url=http://127.0.0.1:8000 \
    --option model=jev-latest --out runs/RSI-Jev-v7.0-VL-4B
python -m decision_index score --edition 0.3 --results results.jsonl.gz --out rescore   # the published file
```

The checkpoint is self-contained (tokenizer, embeddings, the 32 layers, the vision tower and the three heads); nothing is fetched from the base model. The package's `meta.json` turns on the input format it was trained with (`serve_env`); the server applies it at load, so nothing needs to be set. Leave `effort` out of the request and `RSIJEV_EFFORT` unset to get the default this run used.

## Notes

- **How the run was made.** The 0.2.1 suite (150,759 requests) ran with the kit at `87d4650` in seven parts against the same server and checkpoint, and the 2,638 rebuilt GSM8K requests ran under 0.3, as the 0.3 README describes for a resumed 0.2.1 run; all eight parts on 1× RTX PRO 6000 Blackwell. `results.jsonl.gz` is the runner rows of those runs, concatenated and unedited; the old GSM8K and ForecastBench rows are in it and the 0.3 scorer skips them. `environment.json` lists each part's own record.
- **Checkpoint and code.** Every part served the released files (the Hugging Face repo at tag `v7.0-VL`, commit `58baf6fc`; each part checked every file against the release `SHA256SUMS` before starting) with RSI-Jev at `479343b`, the release merge.
- **Deterministic serving.** The server fixes the configurations of six fla kernels instead of autotuning them at startup (every part logged "tune pin: 6 of 6 kernels"); two server processes give bitwise identical outputs on the RTX PRO 6000 and the H200, so a re-run should reproduce these answers.
- **No truncation.** A question longer than 32,768 tokens would get a 422 ("maximum context length") and be recorded as unsupported; none was.
- **Scores.** `score03` on `results.jsonl.gz` (0.3 scoring, the same code as the released 0.3 kit) gives 60.61, `"complete": true`, 140,178 completed. The model's release record reports 60.63 from a run of the same weights served just before release (fp32 tower, no kernel pin); this run is of the released files.
