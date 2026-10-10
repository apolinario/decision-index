# CoCo-Decision-4B 2.1.0

[corners-ai/CoCo-Decision-4B](https://huggingface.co/corners-ai/CoCo-Decision-4B) is a decision model built on `Qwen/Qwen3.5-4B` with LoRA stages merged into the published text-only bf16 weights. The later stages were trained on public train splits of Decision Index source datasets. Version 2.1.0 is read out in one forward pass per question from a readout head over the option-letter tokens, with bidirectional full attention and no generation.

| | |
|---|---|
| Decision Index 0.3 (public) | **53.93** (raw 65.19, breadth skill 51.99) |
| Area skill | knowledge 33.3 · language 63.2 · retrieval 57.1 · tools 75.0 · arts 38.8 |
| Requests | 140,620 requests (140,178 scoreable, 442 excluded), all scoreable `ok`; none unsupported, no errors |
| Results | [corners-ai/CoCo-Decision-4B-decision-index](https://huggingface.co/datasets/corners-ai/CoCo-Decision-4B-decision-index/tree/96c5635ccef0c986b451ef715e8ee34ebfb9efb0/runs/CoCo-Decision-4B-2.1.0) (`--compact`, no suite text) |
| Latency on the run hardware | median 25.6 ms, mean 93.2 ms, p95 299.6 ms (RTX 5090, one sequential client) |

## Running it

The server is `omj serve` from [iamupd/oh-my-jev](https://github.com/iamupd/oh-my-jev) at commit `292dc1b`, semif backend in bf16, no calibration file. The submitted run applied the 2.1.0 LoRA adapter to the public 1.0.0 weights at load time; the merged weights on the Hub gave the same top choice with a maximum logit difference of 0.05 on 3 inputs, and the full suite was not re-run on them.

```sh
git clone https://github.com/iamupd/oh-my-jev && cd oh-my-jev && git checkout 292dc1b
uv sync --extra semif
cat > di.toml <<'TOML'
[backend]
name = "semif"
model = "corners-ai/CoCo-Decision-4B"
revision = "d17348beec75afbba9feb317163e7738a9c9439b"
quant = "bf16"
max_state_tokens = 32768
TOML
OMJ_CONFIG=di.toml uv run omj serve          # http://127.0.0.1:8799/v1/systemone
python -m decision_index pipeline --edition 0.3 --engine http --compact \
    --option base_url=http://127.0.0.1:8799 --option model=CoCo-Decision-4B --out runs/CoCo-Decision-4B
```

That is the fastest path the code supports: one sequential client, no order ensembling, no prefix cache.

## Declared limits

- The server accepts prompts up to 32,768 tokens (`max_state_tokens`), and up to 237 options per question, the option labels that are single tokens in the Qwen3.5 tokenizer.
- Larger requests are refused with HTTP 422 naming the limit.
- In this suite the longest prompt is 22,656 tokens and no question has more than 237 options, so nothing was refused or truncated.

## Suite

Rebuilt with `suite rebuild` and imported with verification: base rows `b2b56d6f…`, added rows `7429f3c9…`, GSM8K rows `75933489…`.

## Training data disclosure

- The second stage trained on the public train splits of the datasets behind 14 suite benchmarks: VAST, iSarcasmEval, When2Call, HellaSwag, WinoGrande, RAGTruth, ContractNLI, New Yorker caption contest, Humicroedit, ANLI, GSM8K, BANKING77, CLINC150 and Amazon ESCI. The first stage also included the GSM8K, BANKING77 and CLINC150 train splits.
- Rows overlapping Decision Index 0.3 evaluation items (13-gram and exact-line matching) were removed before training. No evaluation items were used for training or model selection.
