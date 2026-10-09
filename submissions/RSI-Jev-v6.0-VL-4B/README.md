# RSI-Jev v6.0-VL 4B

[shgao/rsi-jev-v6.0-vl-4b](https://huggingface.co/shgao/rsi-jev-v6.0-vl-4b) is Qwen3.5-4B-Base, fine-tuned, with decision heads after layers 16, 20 and 32: 4.69B parameters including the vision tower, which this suite does not use. It reads the state, the questions and every option in one forward pass and returns a probability for each option; it generates nothing. It chooses its depth: a question is answered at the first of the three layers that is confident enough. This run uses the default effort (unset).

| | |
|---|---|
| Decision Index 0.3 public index | 46.23 (raw 59.17) |
| Requests | 140,178 of 140,178 scoreable, all answered, none unsupported, no errors |
| Median request time | 38.7 ms (one request at a time) |

## Run

```sh
pip install "rsi-jev[fast,vision] @ git+https://github.com/Shanghua-Gao/RSI-Jev@fc982bbd3fb185af2ce629091588cb640400dafb"
rsi-jev serve v6.0-vl-4b --port 8000          # effort unset, as run; 32,768-token and 5,120-option caps; longer is a 422, never cut

git clone https://github.com/apolinario/decision-index && cd decision-index && git checkout 9eb2dbe   # edition 0.3
python -m decision_index pipeline --engine http --option base_url=http://127.0.0.1:8000 \
    --option model=jev-latest --out runs/RSI-Jev-v6.0-VL-4B
python -m decision_index score --edition 0.3 --results results.jsonl.gz --out rescore   # the published file
```

The checkpoint is self-contained (tokenizer, embeddings, the 32 layers and the three heads); nothing is fetched from the base model. Leave `effort` out of the request and `RSIJEV_EFFORT` unset to get the default this run used.

## Notes

- **How the run was made.** The 0.2.1 suite (150,759 requests) ran with the kit at `87d4650` on 1× NVIDIA H200, in seven parts against the same server and checkpoint; the 2,638 rebuilt GSM8K requests ran with the kit at `62d2f51` on 1× RTX PRO 6000 Blackwell, as the 0.3 README describes for a resumed 0.2.1 run. `results.jsonl.gz` is the runner rows of those runs, concatenated and unedited; the old GSM8K and ForecastBench rows are in it and the 0.3 scorer skips them. `environment.json` lists each part's own record.
- **Code.** The run served from RSI-Jev `b61c5a3`; the commands above install current main, `fc982bb`. The serving changes since then apply only to packages that set `serving.pad_multiple`; this one does not, so it is served the same way.
- **No truncation.** A question longer than 32,768 tokens would get a 422 ("maximum context length") and be recorded as unsupported; none was.
- **Hardware.** The server picks a few speed settings from the GPU. On 100 sample requests the RTX PRO 6000 and the H200 gave the same top option on 98, with probabilities within 0.02.
- **Scores.** Re-scoring `results.jsonl.gz` with the kit at `62d2f51` (`score --edition 0.3`; the released edition 0.3, `9eb2dbe`, has identical code) gives 46.23, `"complete": true`, 140,178 completed.
