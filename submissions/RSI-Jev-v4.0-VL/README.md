# RSI-Jev v4.0-VL

[shgao/rsi-jev-v4.0-vl-qwen3.5-2b](https://huggingface.co/shgao/rsi-jev-v4.0-vl-qwen3.5-2b) is a Qwen3.5-2B-Base tower with a cross-attention scorer over the options. It reads the state, the question and every option in one forward pass and returns a probability for each option; it generates nothing. It also takes images, which this suite does not use.

| | |
|---|---|
| Decision Index 0.2.1 | 28.31 (raw 45.29) |
| Decision Index 0.2 | 25.39 (raw 44.03) |
| Requests | 151,229, all answered, none unsupported |
| Median request time | 21 ms (RTX PRO 6000, one request at a time) |

## Run

```sh
pip install "rsi-jev[fast] @ git+https://github.com/Shanghua-Gao/RSI-Jev@83ed4a3b60d21ddea3bc94247e1ca64caa64ad2f"
rsi-jev serve v4.0-vl-2b --port 8000          # 32,768-token and 5,120-option caps; longer is a 422, never cut
python -m decision_index pipeline --engine http --option base_url=http://127.0.0.1:8000 \
    --option model=jev-latest --out runs/RSI-Jev-v4.0-VL
```

The checkpoint and its base model download on first use. About 6 GB of GPU memory holds the model; long multi-question requests need more. A forward pass is capped at 32,768 padded tokens (`RSIJEV_FORWARD_MAX_TOKENS`), so the run fits in one 96 GB card.

## Notes

- **No truncation.** A question longer than 32,768 tokens would get a 422 ("maximum context length") and be recorded as unsupported; none was.
- **Structured criteria** (objects or arrays, as in ChessBench, POP909 and cfcolor) reach the model as compact JSON, the same rendering a structured state gets. The prompt is the same for every benchmark.
- **One resume.** The first pass ran at `cb1246f`, the parent of `83ed4a3` without the per-pass token cap, and ran out of GPU memory on 1,387 requests (ToolRet, API-Bank, BRIGHT). The run was resumed at `83ed4a3`, which retried exactly those rows; the two commits differ only in that cap, which changes how rows are batched, not how they are scored. Their error rows remain in `results.jsonl.gz`, superseded by later rows. The only edit to the file: the internal directory prefix in those rows' `traceback` field is replaced with `<workdir>/`.
