# CoCo-Decision-4B 2.0.0

[corners-ai/CoCo-Decision-4B](https://huggingface.co/corners-ai/CoCo-Decision-4B) is a decision model built on `Qwen/Qwen3.5-4B` with LoRA stages merged into the published text-only bf16 weights. The second stage was trained on public train splits of Decision Index source datasets. It is read out in one forward pass per question from the option-label logits, with no generation.

| | |
|---|---|
| Decision Index 0.3 (public) | **51.93** (raw 63.87, breadth skill 49.77) |
| Area skill | knowledge 31.1 · language 61.5 · retrieval 58.4 · tools 71.1 · arts 32.9 |
| Requests | 140,620 scoreable, all `ok`; none unsupported, no errors |
| Results | [corners-ai/CoCo-Decision-4B-decision-index](https://huggingface.co/datasets/corners-ai/CoCo-Decision-4B-decision-index/tree/bf016abc576707bc5ab8a192f7684afe4879298c/runs/CoCo-Decision-4B-2.0.0) (`--compact`, no suite text) |
| Latency on the run hardware | median 26.1 ms, mean 92.2 ms, p95 302.5 ms (RTX 5090, one sequential client) |

## Running it

The server is `omj serve` from [iamupd/oh-my-jev](https://github.com/iamupd/oh-my-jev) at commit `f0b8c78`, semif backend in bf16, no calibration file.

```sh
git clone https://github.com/iamupd/oh-my-jev && cd oh-my-jev && git checkout f0b8c78
uv sync --extra semif
cat > di.toml <<'TOML'
[backend]
name = "semif"
model = "corners-ai/CoCo-Decision-4B"
revision = "949f1080f3aa58d61481e16456919c81b7b78207"
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
