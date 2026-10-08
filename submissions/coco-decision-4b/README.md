# CoCo-Decision-4B

[corners-ai/CoCo-Decision-4B](https://huggingface.co/corners-ai/CoCo-Decision-4B) is a decision model built on `Qwen/Qwen3.5-4B` with two LoRA stages, both merged into the published text-only bf16 weights. The second stage was trained on public train splits of Decision Index source datasets. It is read out in one forward pass per question from the option-label logits, with no generation.

| | |
|---|---|
| Decision Index 0.3 (public) | **47.80** (raw 60.78, breadth skill 45.92) |
| Area skill | knowledge 28.6 · language 58.5 · retrieval 51.4 · tools 64.5 · arts 32.0 |
| Requests | 140,178 scoreable, all `ok`; none unsupported, no errors |
| Results | [corners-ai/CoCo-Decision-4B-decision-index](https://huggingface.co/datasets/corners-ai/CoCo-Decision-4B-decision-index/tree/64be2c17072cfd853f6f59cdc4448601cd921396/runs/CoCo-Decision-4B) (`--compact`, no suite text) |
| Latency on the run hardware | median 26.1 ms, mean 92.1 ms, p95 302 ms (RTX 5090, one sequential client) |

## Running it

The server is `omj serve` from [iamupd/oh-my-jev](https://github.com/iamupd/oh-my-jev) at commit `f0b8c78`, semif backend in bf16, no calibration file.

```sh
git clone https://github.com/iamupd/oh-my-jev && cd oh-my-jev && git checkout f0b8c78
uv sync --extra semif
cat > di.toml <<'TOML'
[backend]
name = "semif"
model = "corners-ai/CoCo-Decision-4B"
revision = "36a981d44065346b2732e057212bb5c3fc9c5029"
quant = "bf16"
max_state_tokens = 32768
TOML
OMJ_CONFIG=di.toml uv run omj serve          # http://127.0.0.1:8799/v1/systemone
python -m decision_index pipeline --edition 0.3 --engine http --compact \
    --option base_url=http://127.0.0.1:8799 --option model=CoCo-Decision-4B --out runs/CoCo-Decision-4B
```

That is the fastest path the code supports: one sequential client, no order ensembling, no prefix cache.

During the submitted run the second-stage LoRA was applied at load on top of the merged first-stage weights. The Hub `v1.0.0` weights are that LoRA merged in (the 80 merged tensors are bit-identical to PEFT `merge_and_unload`), so outputs can differ only by bf16 rounding.

## Declared limits

- The server accepts prompts up to 32,768 tokens (`max_state_tokens`), and up to 237 options per question, the option labels that are single tokens in the Qwen3.5 tokenizer.
- Larger requests are refused with HTTP 422 naming the limit.
- In this suite the longest prompt is 22,656 tokens and no question has more than 237 options, so nothing was refused or truncated.

## Suite

Rebuilt with `suite rebuild` and imported with verification: base rows `b2b56d6f…`, added rows `7429f3c9…`, GSM8K rows `75933489…`.

## Training data disclosure

- The second stage trained on the public train splits of the datasets behind 14 suite benchmarks: VAST, iSarcasmEval, When2Call, HellaSwag, WinoGrande, RAGTruth, ContractNLI, New Yorker caption contest, Humicroedit, ANLI, GSM8K, BANKING77, CLINC150 and Amazon ESCI. The first stage also included the GSM8K, BANKING77 and CLINC150 train splits.
- Rows overlapping Decision Index 0.3 evaluation items (13-gram and exact-line matching) were removed before training. No evaluation items were used for training or model selection.
